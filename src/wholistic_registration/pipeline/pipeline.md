## WHOLISTIC Registration Pipeline

### Priorities

Things to check (top priority):
- making sure it's CPU/GPU compatible 
- make sure the pipeline can handle single planes?
- why is it going from 0 to larger number? [THIS SHOULD BE CHANGED] 
- need to add option to only register a few frames? and option to downsample in time? [dt = 10, timeRange =0-1000]
- 10 iterations feels like very few? [sanity convergence?], tolerance=1e-3 [alternatives? other criteria to make number of iterations adaptive to how poorly aligned the data] (plot error stats)
- refractor options - options/window size etc should be all in frames, or in all time (both options)


Current issues/questions:
- there are jumps between the references [align references] - Ginny needs to check this
- quality of the reliability mask? - Ginny needs to check this


Other question for the future:
- does the edge-map help? [pause]
- can one remove the noise in the high-frequency data by removing noise in FFT space manually

Second priority:
- aligment to high-resolution reference (this is totally the future - very exciting)
- collect new dataset using new microscope




# Running the pipeline

### Running on Janelia Cluster

Request GPU node
```
bsub -J demo_job -n 2 -gpu "num=2" -q gpu_a100 -Is -W 48:00 /bin/bash
```

Activate environment
```
conda activate wholistic-registration

```
Launch script


```
python /python_packages/wholistic_registration/src/wholistic_registration> python pipeline_vmsr.py
```



---

- Define data path
- Extract metadata from data (pixel resolution, frame rate, data type)
- Convert data to necessary format (if needed)
- Define registration parameters (this should include the version of the code used)
- Save registration parameters to metafile (along with metadata of data)
- Load these parameters from save file 
- Run registration (ideally there should be an estimate of CPU/GPU resources needed based on data volume)
- If there are multiple references, save references into one tif file to inspect visually for drift
- Apply registration to both channels
- Save registered data with metadata from the original file (pixel resolution, frame rate, data type)
- Create downsampled version of registered data (calcium and membrane) to inspect visually results
- Create 4D mask to definie well/poorly registered areas (ie.: what pixels at what time can be trusted)
- 
... 

## Results folder

- registered data
    - calcium
    - membrane
    - references
        - labeled with timepoints that are associated with the reference
    - mask
        - folder of tiffs (one per timepoint)
        - channels: mask

- downsampled data
    - calcium
        - folder of tiffs (one per downsampled timepoint)
        - contains downsampled calcium data
        - channels: original data, registered data, reference used for that timepoint
    - membrane
        - folder of tiffs (one per downsampled timepoint)
        - contains downsampled membrane data
        - channels: original data, registered data, reference used for that timepoint
    - downsampled mask
        - folder of tiffs (one per downsampled timepoint)
        - channels: registered calcium data, registered membrane data, mask (all downsampled data)