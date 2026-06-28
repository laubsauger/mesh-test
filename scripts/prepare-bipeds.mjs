import { mkdir, readdir, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const sourceDir = join(root, 'assets', 'bipeds');
const publicDir = join(root, 'public', 'bipeds');

async function unzip(zipPath) {
  await new Promise((resolve, reject) => {
    const child = spawn('unzip', ['-oq', zipPath, '-d', publicDir], {
      stdio: ['ignore', 'inherit', 'inherit']
    });
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`unzip exited with ${code}: ${zipPath}`));
    });
  });
}

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walk(path));
    if (entry.isFile()) files.push(path);
  }

  return files;
}

await rm(publicDir, { recursive: true, force: true });
await mkdir(publicDir, { recursive: true });

const zipFiles = (await readdir(sourceDir))
  .filter((file) => file.endsWith('.zip'))
  .sort();

if (zipFiles.length === 0) {
  throw new Error(`No Meshy biped zip files found in ${sourceDir}`);
}

for (const zip of zipFiles) {
  await unzip(join(sourceDir, zip));
}

const glbs = (await walk(publicDir)).filter((file) => file.endsWith('.glb'));

// This step only unzips the raw Meshy source into public/bipeds/. The shipped
// assets + manifest are produced by scripts/optimize-bipeds.mjs (npm run
// optimize:bipeds), which merges each character down to one compact GLB.
console.log(`Unzipped ${zipFiles.length} archives -> ${glbs.length} GLBs in public/bipeds/.`);
console.log('Next: npm run optimize:bipeds');
