// Packs a mudslide/Baileys auth folder (creds.json + many small key files) into one JSON blob and back,
// so the whole WhatsApp session can live as a single S3 object. Shared by the Lambda and the local upload script.
import fs from 'node:fs';
import path from 'node:path';

export function packSession(dir) {
  const files = {};
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    if (fs.statSync(full).isFile()) files[name] = fs.readFileSync(full).toString('base64');
  }
  if (!files['creds.json']) throw new Error(`No creds.json in ${dir} - run "npx mudslide login" first`);
  return JSON.stringify({ version: 1, files });
}

export function unpackSession(blob, dir) {
  const { files } = JSON.parse(blob);
  fs.mkdirSync(dir, { recursive: true });
  for (const [name, b64] of Object.entries(files)) {
    // Names come from our own blob, but never let one escape the target folder.
    if (name !== path.basename(name)) throw new Error(`Bad file name in session blob: ${name}`);
    fs.writeFileSync(path.join(dir, name), Buffer.from(b64, 'base64'));
  }
}
