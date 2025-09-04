# PPT-2 Training Scripts

To launch training scripts:

```shell
python src/scripts/train/ppt2/control.py launch control ai2/jupiter-cirrascale-2
python src/scripts/train/ppt2/phase0_launch.py phase0 ai2/jupiter-cirrascale-2  # FIXME
```

Can debug `control` using `dry_run` instead of `launch`.
Eventually, it could be nice to integrate all of these commands.