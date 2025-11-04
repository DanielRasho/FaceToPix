import fs from "fs";
import path from "path";
import axios from "axios";
import * as cheerio from "cheerio";

// 🧩 CONFIG
const urlsFile = "./pixie.json"; // file containing ["url1", "url2", ...]
const outputDir = "./src2"; // where images will be saved
const batchSize = 20; // number of images per batch
const sleepMs = 1000; // pause between batches (ms)

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
    const { data } = await axios.request({
      url: "https://pixie.haus/" + pageUrl,
      headers: {
        Accept:
          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.5",
        "User-Agent":
          "Mozilla/5.0 (X11; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        Connection: "keep-alive",
        // 'Cookie': 'your-cookie-here' // if needed
      },
    });
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
      console.log(
        `😴 Sleeping for ${sleepMs / 1000}s after ${i + 1} downloads...`
      );
      await sleep(sleepMs);
    }
  }

  console.log("🎉 All downloads complete!");
}

main();
