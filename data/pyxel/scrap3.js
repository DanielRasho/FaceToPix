import axios from "axios";
import * as cheerio from "cheerio";
import fs from "fs";
import path from "path";

async function scrapeAndDownloadImages(startPage, endPage, outputFolder) {
    // Create output folder if it doesn't exist
    if (!fs.existsSync(outputFolder)) {
        fs.mkdirSync(outputFolder, { recursive: true });
        console.log(`Created output folder: ${outputFolder}\n`);
    }

    // Create axios instance with session support
    const client = axios.create({
        baseURL: 'https://pixeljoint.com',
        headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        },
        withCredentials: true
    });

    try {
        // First, make the initial search request to establish the session
        const searchUrl = '/pixels/new_icons.asp?search=&dimo=%3E%3D&dim=48&colorso=%3E%3D&colors=2&tran=&anim=&iso=&av=&owner=&d=&dosearch=1&ob=search&action=search';
        console.log('Establishing search session...\n');
        await client.get(searchUrl);

        let totalDownloaded = 0;

        // Loop through the pages
        for (let page = startPage; page <= endPage; page++) {
            console.log(`--- Page ${page} ---`);
            
            const pageUrl = `/pixels/new_icons.asp?q=1&pg=${page}`;
            const response = await client.get(pageUrl);
            
            const $ = cheerio.load(response.data);
            
            // Find all divs with class "bx" and extract img src
            const imageSrcs = [];
            $('div.bx img').each((i, elem) => {
                const src = $(elem).attr('src');
                if (src) {
                    imageSrcs.push(src);
                }
            });
            
            console.log(`Found ${imageSrcs.length} images on page ${page}`);
            
            // Download each image
            for (let i = 0; i < imageSrcs.length; i++) {
                const src = imageSrcs[i];
                const fullUrl = `https://pixeljoint.com${src}`;
                
                // Extract filename from src or create unique name
                const filename = src.split('/').pop() || `page${page}_image${i + 1}.png`;
                const filepath = path.join(outputFolder, filename);
                
                try {
                    await downloadImage(fullUrl, filepath);
                    console.log(`  ✓ Downloaded: ${filename}`);
                    totalDownloaded++;
                } catch (error) {
                    console.error(`  ✗ Failed to download ${filename}:`, error.message);
                }
                
                // Small delay between downloads
                await new Promise(resolve => setTimeout(resolve, 200));
            }
            
            console.log('');
            
            // Delay between pages
            if (page < endPage) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        
        console.log(`\n=== Complete ===`);
        console.log(`Total images downloaded: ${totalDownloaded}`);
        console.log(`Saved to: ${path.resolve(outputFolder)}`);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

async function downloadImage(url, filepath) {
    const response = await axios.get(url, {
        responseType: 'arraybuffer',
        headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    });
    
    fs.writeFileSync(filepath, response.data);
}

// Usage: scrape pages 1-5 and save to 'images' folder
scrapeAndDownloadImages(115, 208, './src3');