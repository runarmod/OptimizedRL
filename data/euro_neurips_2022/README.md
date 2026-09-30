# EURO Meets NeurIPS 2022 instances (subset)

30 of the 250 public VRPTW instances from the
[EURO Meets NeurIPS 2022 Vehicle Routing Competition quickstart](https://github.com/ortec/euro-neurips-vrp-2022-quickstart)
(`instances/`), copied unchanged. The DVSP environment reads them as static
instances to sample dynamic request streams from.

These are the 10 train, 10 validation and 10 test instances listed in
`config_dvsp.yaml`. They were chosen by shuffling the 250 sorted file names
with `split_seed: 0`, splitting 34% / 34% / 32% and taking the first 10 of
each part (see `split_instances` in `src/dvsp/instance.py`), the same
procedure as paper 02 (Hoppe et al., 2025) but with numpy's random generator.

Data by ORTEC, licensed under
[CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/); see `LICENSE`.
