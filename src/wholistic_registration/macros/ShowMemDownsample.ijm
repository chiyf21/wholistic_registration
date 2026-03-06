// =====================================================
// Stable 3-Channel HyperStack Builder (Frame Parsed)
// raw_membrane | membrane | reference (interval-based)
// =====================================================

setBatchMode(true);

// -----------------------------
// Select root directory
// -----------------------------
rootDir = getDirectory("Select root folder");

rawDir = rootDir + "raw_membrane/";
memDir = rootDir + "membrane/";
refDir = rootDir + "reference/";

// -----------------------------
// Get file lists
// -----------------------------
rawList = getFileList(rawDir);
memList = getFileList(memDir);
refList = getFileList(refDir);

rawFiles = newArray();
memFiles = newArray();
refFiles = newArray();

for (i = 0; i < rawList.length; i++)
    if (endsWith(rawList[i], ".tif"))
        rawFiles = Array.concat(rawFiles, rawList[i]);

for (i = 0; i < memList.length; i++)
    if (endsWith(memList[i], ".tif"))
        memFiles = Array.concat(memFiles, memList[i]);

for (i = 0; i < refList.length; i++)
    if (endsWith(refList[i], ".tif"))
        refFiles = Array.concat(refFiles, refList[i]);

Array.sort(rawFiles);
Array.sort(memFiles);
Array.sort(refFiles);

if (rawFiles.length != memFiles.length)
    exit("raw/membrane mismatch");

nFrames = rawFiles.length;

// --------------------------------------------------
// Parse TRUE frame numbers from membrane filenames
// --------------------------------------------------
frameNumber = newArray(nFrames);

for (i = 0; i < nFrames; i++) {

    name = memFiles[i];
    name = replace(name, ".tif", "");

    parts = split(name, "_");
    lastPart = parts[parts.length-1];

    frameNumber[i] = parseInt(lastPart);
}

// -----------------------------
// Parse reference intervals
// -----------------------------
refStart = newArray(refFiles.length);
refEnd   = newArray(refFiles.length);

for (i = 0; i < refFiles.length; i++) {

    name = refFiles[i];
    name = replace(name, "vol_ref_", "");
    name = replace(name, ".tif", "");

    parts = split(name, "_");

    refStart[i] = parseInt(parts[0]);
    refEnd[i]   = parseInt(parts[1]);
}

// -----------------------------
// Inspect first file
// -----------------------------
open(rawDir + rawFiles[0]);
getDimensions(w, h, c, zDepth, tDepth);
close();

nChannels = 3;
nSlice = zDepth;

// -----------------------------
// Create hyperstack
// -----------------------------
newImage("Combined", "16-bit black",
         w, h, nChannels*nSlice*nFrames);

run("Stack to Hyperstack...",
    "order=xyczt(default) channels=3 slices=" + nSlice +
    " frames=" + nFrames +
    " display=Composite");

selectWindow("Combined");

// -----------------------------
// Main loop
// -----------------------------
for (f = 0; f < nFrames; f++) {

    trueFrame = frameNumber[f];

    // -------- Channel 1: raw_membrane --------
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
    selectWindow(rawTitle); close();


    // -------- Channel 2: membrane --------
    open(memDir + memFiles[f]);
    memTitle = getTitle();

    for (z = 1; z <= nSlice; z++) {
        selectWindow(memTitle);
        setSlice(z);
        run("Copy");

        selectWindow("Combined");
        Stack.setPosition(2, z, f+1);
        run("Paste");
    }
    selectWindow(memTitle); close();


    // -------- Channel 3: reference --------
    refIndex = -1;

    for (r = 0; r < refFiles.length; r++) {
        if (trueFrame >= refStart[r] && trueFrame <= refEnd[r]) {
            refIndex = r;
            break;
        }
    }

    if (refIndex != -1) {

        open(refDir + refFiles[refIndex]);
        refTitle = getTitle();

        for (z = 1; z <= nSlice; z++) {
            selectWindow(refTitle);
            setSlice(z);
            run("Copy");

            selectWindow("Combined");
            Stack.setPosition(3, z, f+1);
            run("Paste");
        }

        selectWindow(refTitle); close();
    }

    if (f % 50 == 0)
        run("Collect Garbage");

    showProgress(f+1, nFrames);
}

setBatchMode(false);

// Auto contrast
for (c = 1; c <= 3; c++) {
    Stack.setChannel(c);
    run("Enhance Contrast", "saturated=0.35");
}

print("Done. Loaded " + nFrames + " frames with reference.");