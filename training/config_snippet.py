360:class LeRobotStrawberryDataConfig(DataConfigFactory):
361-    """Custom data config for UR5e strawberry picking in Isaac Sim."""
362-
363-    @override
364-    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
365-        repack_transform = _transforms.Group(
366-            inputs=[
367-                _transforms.RepackTransform(
368-                    {
369-                        "observation/cam1": "observation.images.cam1",
370-                        "observation/cam2": "observation.images.cam2",
371-                        "observation/state": "observation.state",
372-                        "actions": "action",
373-                        "prompt": "task",
374-                    }
375-                )
376-            ]
377-        )
378-        data_transforms = _transforms.Group(
379-            inputs=[strawberry_policy.StrawberryInputs(model_type=model_config.model_type)],
380-            outputs=[strawberry_policy.StrawberryOutputs()],
381-        )
382-        delta_action_mask = _transforms.make_bool_mask(6, -2)
383-        data_transforms = data_transforms.push(
384-            inputs=[_transforms.DeltaActions(delta_action_mask)],
385-            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
386-        )
387-        model_transforms = ModelTransformFactory()(model_config)
388-        return dataclasses.replace(
389-            self.create_base_config(assets_dirs, model_config),
390-            action_sequence_keys=("action",),
391-            repack_transforms=repack_transform,
392-            data_transforms=data_transforms,
393-            model_transforms=model_transforms,
394-        )
395-

1014:        name="pi05_strawberry",
1015-        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=10),
1016-        data=LeRobotStrawberryDataConfig(
1017-            repo_id="local/strawberry_picking",
1018-        ),
1019-        batch_size=4,
1020-        lr_schedule=_optimizer.CosineDecaySchedule(
1021-            warmup_steps=500,
1022-            peak_lr=5e-5,
1023-            decay_steps=100_000,
1024-            decay_lr=5e-5,
1025-        ),
1026-        freeze_filter=pi0_config.Pi0Config(
1027-            paligemma_variant="gemma_2b_lora",
1028-            action_expert_variant="gemma_300m_lora",
1029-        ).get_freeze_filter(),
1030-        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
1031-        ema_decay=None,
1032-        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
1033-        num_train_steps=30_000,
1034-        wandb_enabled=False,
1035-    ),
1036-    *polaris_config.get_polaris_configs(),
1037-]
1038-
1039-if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
