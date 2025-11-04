import fs from "fs";
import path from "path";
import axios from "axios";

// 🧩 CONFIG
const urlsFile = "./pixie.json";   // JSON file containing ["https://...jpg", "https://...png", ...]
const outputDir = "./src2";        // folder where images will be saved
const batchSize = 20;              // number of images per batch
const sleepMs = 1000;              // pause between batches (ms)

// Ensure output directory exists
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Sleep helper
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function downloadImage(url, filename) {
  try {
    const response = await axios({
      url,
      responseType: "arraybuffer",
      headers: {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
      },
    });

    fs.writeFileSync(filename, response.data);
    console.log(`✅ Saved: ${filename}`);
  } catch (err) {
    console.error(`❌ Failed to download ${url}: ${err.message}`);
  }
}

async function main() {
  // Read and parse JSON
  const raw = fs.readFileSync(urlsFile, "utf-8");
  const urls = JSON.parse(raw);

  if (!Array.isArray(urls)) {
    throw new Error("❌ The JSON file must contain an array of image URLs.");
  }

  console.log(`📋 Loaded ${urls.length} image URLs from ${urlsFile}`);

  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];
    const filename = path.join(outputDir, path.basename(new URL(url).pathname));

    await downloadImage(url, filename);

    // Batch sleep control
    if ((i + 1) % batchSize === 0 && i + 1 < urls.length) {
      console.log(`😴 Sleeping for ${sleepMs / 1000}s after ${i + 1} downloads...`);
      await sleep(sleepMs);
    }
  }

  console.log("🎉 All downloads complete!");
}

main();