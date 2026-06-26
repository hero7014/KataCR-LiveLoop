import orbax.checkpoint as ocp
from orbax.checkpoint import type_handlers as ocp_type_handlers
from pathlib import Path
import shutil
import jax
from flax.training import train_state


_orig_deserialize_sharding = ocp_type_handlers._deserialize_sharding_from_json_string


def _deserialize_sharding_with_cpu_fallback(sharding_string: str):
  try:
    return _orig_deserialize_sharding(sharding_string)
  except ValueError as e:
    msg = str(e)
    if 'SingleDeviceSharding' in msg and 'was not found in jax.local_devices()' in msg:
      devices = jax.local_devices()
      if not devices:
        raise
      print(f"Checkpoint device mapping fallback: restoring saved sharding onto local device {devices[0]}")
      return jax.sharding.SingleDeviceSharding(devices[0])
    raise


ocp_type_handlers._deserialize_sharding_from_json_string = _deserialize_sharding_with_cpu_fallback

class CheckpointManager(ocp.CheckpointManager):
  def __init__(self, path_save, max_to_keep=1, remove_old=False):
    self.path_save = path_save = str(Path(path_save).resolve())
    if remove_old:
      shutil.rmtree(path_save, ignore_errors=True)
    super().__init__(
      path_save,
      options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, step_format_fixed_length=3),
      item_names={'variables', 'config'},
      item_handlers={'variables': ocp.StandardCheckpointHandler(), 'config': ocp.JsonCheckpointHandler()}
    )
  
  def save(self, epoch: int, state: train_state.TrainState, config: dict, verbose: bool = True):
    args = ocp.args
    for k, v in config.items():
      if isinstance(v, Path): config[k] = str(v)
    config['_step'] = int(state.step)
    if verbose:
      print(f"Save weights at {self.path_save}/{epoch:03}/")
    variables = {'params': state.params}
    if hasattr(state, 'batch_stats'): variables.update({'batch_stats': state.batch_stats})
    return super().save(epoch, args=args.Composite(
      variables=args.StandardSave(variables),
      config=args.JsonSave(config)
    ))
  
  def restore(self, epoch: int):
    return super().restore(epoch)
  
  def load(self, state: train_state.TrainState, epoch: int, need_opt: bool = False):
    ret = self.restore(epoch)
    state = state.replace(step=ret['config']['_step'], params=ret['params'])
    return state
