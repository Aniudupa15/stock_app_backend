// After `npm run login` (QR scan), pack the local session folder and upload it to S3, then delete
// the local copy - two clients using the same keys would drift apart and get the device logged out.
//   node upload-session.mjs <bucket> [region]
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { packSession } from './session-blob.mjs';

const [bucket, region = 'ap-south-1'] = process.argv.slice(2);
if (!bucket) {
  console.error('usage: node upload-session.mjs <bucket> [region]');
  process.exit(1);
}
const sessionDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '.session');
const tmpFile = path.join(os.tmpdir(), `wa-session-${process.pid}.json`);
fs.writeFileSync(tmpFile, packSession(sessionDir), { mode: 0o600 });
try {
  execFileSync('aws', ['s3', 'cp', tmpFile, `s3://${bucket}/session.json`, '--region', region], { stdio: 'inherit', shell: process.platform === 'win32' });
} finally {
  fs.rmSync(tmpFile, { force: true });
}
fs.rmSync(sessionDir, { recursive: true, force: true });
console.log('Session uploaded to S3 and local copy deleted.');
