Introduction
============

OLMo-core represents a major rewrite of the original training and modeling code from `OLMo <https://github.com/allenai/OLMo>`_
with a focus on performance and API stability.
It aims to provide a standard set of robust tools that can be used by LLM researchers at `AI2 <https://allenai.org>`_ and other organizations
to build their research projects on.

The library is centered around a highly efficient, yet flexible, :class:`~olmo_core.train.Trainer` and a :mod:`~olmo_core.launch`
module that handles all of the boilerplate of launching experiments on `Beaker <https://beaker.org>`_
or other platforms. It also comes with a simple, yet optimized, :class:`~olmo_core.nn.transformer.Transformer`
model and many other useful :class:`torch.nn.Module` implementations.

Most users will likely follow a workflow that looks like this:

1. Define the various components of an experiment through configuration classes.
   For example::

     model_config = TransformerConfig.llama2_7B(...)
     optim_config = AdamWConfig(lr=1e-3, ...)
     data_config = MemMapDatasetConfig(...)
     trainer_config = TrainerConfig(...)

2. Build the corresponding components within a ``main()`` function at runtime and then call :meth:`Trainer.fit() <olmo_core.train.Trainer.fit>`.
   For example::

     def main(model_config, optim_config, data_config, trainer_config):
         model = model_config.build()
         optim = optim_config.build()
         dataset = data_config.build()
         trainer = trainer_config.build(model, optim, dataset)

         trainer.fit()

     if __name__ == "__main__":
         prepare_training_environment(seed=SEED)
         try:
             main()
         finally:
             teardown_training_environment()

3. Launch their training script with a :mod:`~olmo_core.launch` config, like the :class:`~olmo_core.launch.beaker.BeakerLaunchConfig`.
   For example::

     launch_config = BeakerLaunchConfig(...)
     launch_config.launch(follow=True)

You can find a complete example of this workflow in the `Train a language model <examples/train.rst>`_ example.
