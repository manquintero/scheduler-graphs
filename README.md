# Scheduler Charts


## Generate Data

```sh
    for i in {1..30}; do
        stress-ng --all 0 --class cpu -t 60s --metrics-brief |& tee ~/${i}.txt;
    done
```


## Stress-ng version

`stress-ng 0.17.04 g2f22ad595f06`