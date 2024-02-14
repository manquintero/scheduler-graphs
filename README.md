# Scheduler Charts


## Generate Data

```sh
    for i in {1..30}; do
        stress-ng --all 0 --class cpu -t 60s --metrics-brief |& tee "/tmp/${i}.txt";
    done
```
