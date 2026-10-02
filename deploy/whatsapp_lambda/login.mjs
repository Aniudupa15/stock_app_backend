// Link WhatsApp once on this PC: shows a QR code (WhatsApp > Linked devices > Link a device).
// The session lands in ./.session; then run upload-session.mjs to move it to S3.
//   node login.mjs            (QR code)
//   node login.mjs --pairing-code
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const result = spawnSync(process.execPath, [path.join(here, 'node_modules', 'mudslide', 'build', 'index.js'), 'login', ...process.argv.slice(2)], {
  stdio: 'inherit',
  env: { ...process.env, MUDSLIDE_CACHE_FOLDER: path.join(here, '.session') },
});
process.exit(result.status ?? 1);
