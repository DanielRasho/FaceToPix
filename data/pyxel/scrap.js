import fs from "fs";
import path from "path";
import axios from "axios";
import * as cheerio from "cheerio";

// 🧩 CONFIG
const urlsFile = "./src.json";         // file containing ["url1", "url2", ...]
const outputDir = "./downloads";        // where images will be saved
const batchSize = 5;                    // number of images per batch
const sleepMs = 3000;                  // pause between batches (ms)

// Ensure output directory exists
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir);
}

// Sleep helper
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

    const src = $("img.fullsize").first().attr("src");
    if (!src) {
      console.warn(`⚠️ No <img class="fullsize"> found at ${pageUrl}`);
      return;
    }

    const imageUrl = new URL(src, pageUrl).href;
    const filename = path.join(
      outputDir,
      path.basename(new URL(imageUrl).pathname)
    );

    await downloadImage(imageUrl, filename);
  } catch (err) {
    console.error(`❌ Failed to process ${pageUrl}: ${err.message}`);
  }
}

async function main() {
  // Read and parse JSON
  const raw = fs.readFileSync(urlsFile, "utf-8");
  const urls = JSON.parse(raw);
  if (!Array.isArray(urls)) {
    throw new Error("❌ The JSON file must contain an array of URLs.");
  }

  console.log(`📋 Loaded ${urls.length} URLs from ${urlsFile}`);

  for (let i = 0; i < urls.length; i++) {
    await processUrl(urls[i]);

    if ((i + 1) % batchSize === 0 && i + 1 < urls.length) {
      console.log(`😴 Sleeping for ${sleepMs / 1000}s after ${i + 1} downloads...`);
      await sleep(sleepMs);
    }
  }

  console.log("🎉 All downloads complete!");
}

main();