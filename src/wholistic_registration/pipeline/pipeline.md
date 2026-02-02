## WHOLISTIC Registration Pipeline

### Priorities


260130
- Check single channel works
- Check timewindow option words
- Change downsample function to be more efficient - modularize - downsample each channel independently
- Run reliability mask on f2013 - check the mask for false positives and false negatives?
- Make Fiji macro (I can do this) to load data as multiple color channels


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
https://hhmi.atlassian.net/wiki/spaces/SCS/pages/152469552/Janelia+Compute+Cluster#GPU-Compute-Cluster

Request GPU node
```
bsub -J demo_job -n 10 -gpu "num=1" -q gpu_a100 -Is -W 12:00 /bin/bash
```


Activate environment
```
conda activate wholistic-registration

```
Launch script


```
python /groups/ahrens/home/ruttenv/python_packages/wholistic_registration/src/wholistic_registration/pipeline_vmsr.py
```

Queue	RAM per slot	Slots per GPU	Total RAM per GPU
gpu_tesla_large	30 GB	12	360 GB
gpu_rtx	18 GB	5	90 GB
gpu_tesla	15 GB	5	75 GB
gpu_any / gpu_short	15 GB	2	30 GB


1. bjobs
2. jobs -l <jobid> # check which GPU you are using
3. https://lsf-rtm.int.janelia.org/cacti/plugins/grid/grid_bjobs.php?action=viewlist&reset=1&clusterid=1&job_user=ruettenv&status=RUNNING&page=1


### Notes on memory requirements
- GPU load mostly maxed out when computing the references
- CPU load maxed out when loading an entire chunk


```
nvidia-smi && echo "---" && free -h && echo "---" && echo "CPUs: $(nproc)"
```

To monitor VRAM usage in real-time while your job runs:
```watch -n 1 nvidia-smi```

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