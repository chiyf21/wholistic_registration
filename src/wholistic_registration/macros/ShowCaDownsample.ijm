// =====================================================
// Stable Dual-Channel HyperStack Builder
// No Concatenate, No Merge Channels
// =====================================================

setBatchMode(true);

// -----------------------------
// Select directories
// -----------------------------
rootDir = getDirectory("Select root folder");

rawDir = rootDir + "raw_calcium/";
calDir = rootDir + "calcium/";

// -----------------------------
// Get file lists
// -----------------------------
rawList = getFileList(rawDir);
calList = getFileList(calDir);

rawFiles = newArray();
calFiles = newArray();

for (i = 0; i < rawList.length; i++) {
    if (endsWith(toLowerCase(rawList[i]), ".tif") ||
        endsWith(toLowerCase(rawList[i]), ".tiff"))
        rawFiles = Array.concat(rawFiles, rawList[i]);
}

for (i = 0; i < calList.length; i++) {
    if (endsWith(toLowerCase(calList[i]), ".tif") ||
        endsWith(toLowerCase(calList[i]), ".tiff"))
        calFiles = Array.concat(calFiles, calList[i]);
}

if (rawFiles.length == 0)
    exit("No TIFF files found.");

if (rawFiles.length != calFiles.length)
    exit("File number mismatch.");

Array.sort(rawFiles);
Array.sort(calFiles);

nFrames = rawFiles.length;

// -----------------------------
// Inspect first file
// -----------------------------
open(rawDir + rawFiles[0]);
getDimensions(w, h, c, zDepth, tDepth);
//bitDepth = bitDepth();
close();

nChannels = 2;
nSlice   = zDepth;

// -----------------------------
// Create final HyperStack
// -----------------------------
newImage("Combined", "16-bit black", w, h, nChannels*nSlice*nFrames);
run("Stack to Hyperstack...",
    "order=xyczt(default) channels=" + nChannels +
    " slices=" + nSlice +
    " frames=" + nFrames +
    " display=Composite");

selectWindow("Combined");

// -----------------------------
// Fill data
// -----------------------------
for (f = 0; f < nFrames; f++) {

    // -------- Channel 1 --------
    open(rawDir + rawFiles[f]);
    rawTitle = getTitle();

    for (z = 1; z <= nSlice; z++) {

        selectWindow(rawTitle);
        setSlice(z);
        run("Copy");

        selectWindow("Combined");
        Stack.setPosition(1, z, f+1);
        run("Paste");
    }

    selectWindow(rawTitle);
    close();

    // -------- Channel 2 --------
    open(calDir + calFiles[f]);
    calTitle = getTitle();

    for (z = 1; z <= nSlice; z++) {

        selectWindow(calTitle);
        setSlice(z);
        run("Copy");

        selectWindow("Combined");
        Stack.setPosition(2, z, f+1);
        run("Paste");
    }

    selectWindow(calTitle);
    close();

    if (f % 50 == 0)
        run("Collect Garbage");

    showProgress(f+1, nFrames);
}

setBatchMode(false);

// -----------------------------
// Auto contrast
// -----------------------------
for (c = 1; c <= nChannels; c++) {
    Stack.setChannel(c);
    run("Enhance Contrast", "saturated=0.35");
}

print("Done. Loaded " + nFrames + " frames.");