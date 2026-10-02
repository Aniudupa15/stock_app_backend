// Lambda Function URL handler: POST {"to": "me" | "<number>" | "<group-jid>", "message": "..."}
// with header x-api-key: <API_SECRET>. Restores the WhatsApp session from S3 into /tmp, runs
// `mudslide send`, then writes the session back (Baileys rotates keys on every connection, so a
// stale copy would eventually get the linked device logged out).
// An S3 lock object (see acquireLock) keeps two sends from racing on the same session object.
import { S3Client, GetObjectCommand, PutObjectCommand, HeadObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3';
import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { packSession, unpackSession } from './session-blob.mjs';

const BUCKET = process.env.SESSION_BUCKET;
const KEY = process.env.SESSION_KEY || 'session.json';
const API_SECRET = process.env.API_SECRET || '';
const MUDSLIDE = path.join(path.dirname(fileURLToPath(import.meta.url)), 'node_modules', 'mudslide', 'build', 'index.js');
const MAX_MESSAGE_CHARS = 4000;
// mudslide's own command timeout is a fixed 60s: its --timeout flag is broken in 0.38.1 (commander calls
// parseInt(value, previousDefault) -> radix 60 -> NaN -> the watchdog fires instantly). Lambda timeout is 90s
// so the session still gets saved after a slow send.

const LOCK_KEY = 'session.lock';
const LOCK_STALE_MS = 120_000; // > Lambda timeout, so a lock left by a crashed invocation expires on its own.

const s3 = new S3Client({});

// New accounts can't reserve concurrency (account minimum of 10 unreserved), so serialise sends with an
// S3 conditional write instead: only one invocation can create the lock object; the rest get a 429.
async function acquireLock() {
  const put = () => s3.send(new PutObjectCommand({ Bucket: BUCKET, Key: LOCK_KEY, Body: String(Date.now()), IfNoneMatch: '*' }));
  try {
    await put();
    return true;
  } catch (err) {
    if (err.$metadata?.httpStatusCode !== 412 && err.name !== 'PreconditionFailed') throw err;
  }
  const head = await s3.send(new HeadObjectCommand({ Bucket: BUCKET, Key: LOCK_KEY })).catch(() => null);
  if (head && Date.now() - head.LastModified.getTime() < LOCK_STALE_MS) return false;
  await s3.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: LOCK_KEY })).catch(() => {});
  try {
    await put();
    return true;
  } catch {
    return false;
  }
}

const reply = (statusCode, body) => ({
  statusCode,
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
});

function authorized(headers) {
  const given = Buffer.from(headers['x-api-key'] || '');
  const expected = Buffer.from(API_SECRET);
  return expected.length > 0 && given.length === expected.length && crypto.timingSafeEqual(given, expected);
}

function runMudslide(args, cacheDir) {
  return new Promise((resolve) => {
    const proc = spawn(process.execPath, [MUDSLIDE, ...args], {
      env: { ...process.env, MUDSLIDE_CACHE_FOLDER: cacheDir, HOME: '/tmp' },
    });
    let output = '';
    proc.stdout.on('data', (d) => { output += d; });
    proc.stderr.on('data', (d) => { output += d; });
    proc.on('close', (code) => resolve({ code, output }));
  });
}

export const handler = async (event) => {
  const headers = Object.fromEntries(Object.entries(event.headers || {}).map(([k, v]) => [k.toLowerCase(), v]));
  if (!authorized(headers)) return reply(401, { success: false, error: 'unauthorized' });

  let body;
  try {
    const raw = event.isBase64Encoded ? Buffer.from(event.body || '', 'base64').toString() : event.body || '{}';
    body = JSON.parse(raw);
  } catch {
    return reply(400, { success: false, error: 'body must be JSON' });
  }
  const to = String(body.to || 'me').trim();
  const message = typeof body.message === 'string' ? body.message : '';
  if (!message.trim()) return reply(400, { success: false, error: 'message is required' });
  if (message.length > MAX_MESSAGE_CHARS) return reply(400, { success: false, error: `message over ${MAX_MESSAGE_CHARS} chars` });
  if (!/^(me|\d{8,15}|[\d-]+@g\.us)$/.test(to)) return reply(400, { success: false, error: 'to must be "me", a number with country code, or a group JID' });

  if (!(await acquireLock())) return reply(429, { success: false, error: 'busy - another send is in progress, retry shortly' });
  const cacheDir = fs.mkdtempSync('/tmp/mudslide-');
  try {
    let blob;
    try {
      const obj = await s3.send(new GetObjectCommand({ Bucket: BUCKET, Key: KEY }));
      blob = await obj.Body.transformToString();
    } catch (err) {
      if (err.name === 'NoSuchKey') return reply(503, { success: false, error: 'no WhatsApp session uploaded yet' });
      throw err;
    }
    unpackSession(blob, cacheDir);

    const { code, output } = await runMudslide(['send', to, message], cacheDir);

    // Save whatever key state Baileys ended up with, even on failure - it may have rotated keys before failing.
    if (fs.existsSync(path.join(cacheDir, 'creds.json'))) {
      await s3.send(new PutObjectCommand({ Bucket: BUCKET, Key: KEY, Body: packSession(cacheDir), ContentType: 'application/json' }));
    }

    const sent = code === 0 && output.includes('Done');
    console.log(JSON.stringify({ to, chars: message.length, code, sent, output: output.slice(-2000) }));
    if (sent) return reply(200, { success: true });
    const loggedOut = /logged ?out|Not logged in/i.test(output);
    return reply(502, {
      success: false,
      error: loggedOut ? 'whatsapp_logged_out - re-run login and upload-session' : 'send_failed',
      detail: output.slice(-500),
    });
  } finally {
    fs.rmSync(cacheDir, { recursive: true, force: true });
    await s3.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: LOCK_KEY })).catch(() => {});
  }
};
