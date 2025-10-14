import fs from "fs";
import path from "path";
import axios from "axios";
import * as cheerio from "cheerio";

// 🧩 CONFIG
const urlsFile = "./urls.txt";          // file containing list of URLs (one per line)
const outputDir = "./downloads";        // where images will be saved
const batchSize = 5;                    // number of images to download before sleeping
const sleepMs = 10000;                  // how long to sleep between batches (ms)

// Ensure output directory exists
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir);
}

// Utility sleep
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function downloadImage(url, filename) {
  const response = await axios({
    url,
    responseType: "arraybuffer",
  });
  fs.writeFileSync(filename, response.data);
  console.log(`✅ Saved: ${filename}`);
}

async function processUrl(pageUrl) {
  try {
    const { data } = await axios.get(pageUrl);
    const $ = cheerio.load(data);

    // Extract first fullsize image
    const src = $("img.fullsize").first().attr("src");
    if (!src) {
      console.warn(`⚠️ No fullsize image found at ${pageUrl}`);
      return;
    }

    // Resolve absolute URL
    const imageUrl = new URL(src, pageUrl).href;
    const filename = path.join(
      outputDir,
      path.basename(new URL(imageUrl).pathname)
    );

    await downloadImage(imageUrl, filename);
  } catch (err) {
    console.error(`❌ Failed to process ${pageUrl}:`, err.message);
  }
}

async function main() {
  // Read URLs from file
  const urls = fs.readFileSync(urlsFile, "utf-8")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  console.log(`📋 Loaded ${urls.length} URLs from ${urlsFile}`);

  for (let i = 0; i < urls.length; i++) {
    await processUrl(urls[i]);

    // Pause every batch
    if ((i + 1) % batchSize === 0 && i + 1 < urls.length) {
      console.log(`😴 Sleeping ${sleepMs / 1000}s after ${i + 1} downloads...`);
      await sleep(sleepMs);
    }
  }

  console.log("🎉 All done!");
}

main();