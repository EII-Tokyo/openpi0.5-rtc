"""22000-step deployment: apply the training run's 224px model/transforms."""
import dataclasses as dc
import logging
import serve_policy
from openpi.training import config

NAME = 'eii_with_rinse_cam4_fullft_from_april_original_38repo'
CHECKPOINT = '/app/checkpoints/' + NAME + '/6beba2e5-e00a-431f-b8a8-b95bf48ac168/22000'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    original = config.get_config(NAME)
    config._CONFIGS_DICT[NAME] = dc.replace(
        original,
        model=dc.replace(original.model, image_resolution=(224, 224)),
        data=dc.replace(original.data, transform_pipeline=dc.replace(
            original.data.transform_pipeline, image_resolution=(224, 224))),
    )
    logging.info('Deployment checkpoint=%s resolution=224x224 cameras=4', CHECKPOINT)
    serve_policy.main(serve_policy.Args(
        warmup_subtask=False,
        policy=serve_policy.Checkpoint(config=NAME, dir=CHECKPOINT),
    ))
