"""Model definition for MMM."""

import numpy as np
import pandas as pd

from mmm_utils.modeling import MMM, MMMConfig, MediaTransformSpec
from mmm_utils.modeling.prior import PriorSpec
from mmm_utils.modeling import Interaction, BetaPriors

# pylint: disable=missing-function-docstring


# ——————————————————————————————————————————————————————————————————————————— #
#
# ——————————————————————————————————————————————————————————————————————————— #
def media_transforms(half_life=0.5, sigma=0.1, alpha_prior=False, saturation="Hill"):
    # half_life = -log(2) / log(mu)
    mu = np.exp(-np.log(2) / half_life)
    l_max = int(-np.log(100) / np.log(mu))
    adstock_params = {
        "l_max": np.clip(l_max, a_min=8, a_max=None),
        "normalize": False,
    }

    if saturation == "Hill":
        saturation_params = {
            "k": PriorSpec("Beta", {"alpha": 2.0, "beta": 2.0}),
            "n": PriorSpec("Gamma", {"mu": 2.0, "sigma": 1.0}),
        }
    elif saturation == "Logistic":
        saturation_params = {
            "lam": PriorSpec(
                "TruncatedNormal",
                {"mu": 1.0, "sigma": 0.5, "lower": 0.0, "upper": 5.0},
            )
        }
    else:
        raise ValueError(f"Unknown saturation function: {saturation}")

    return MediaTransformSpec(
        adstock="Geometric",  # Delayed",
        adstock_params=adstock_params
        if alpha_prior
        else adstock_params | {"alpha": mu},
        adstock_priors={
            "alpha": PriorSpec(
                "TruncatedNormal",
                {"mu": mu, "sigma": sigma, "lower": 0.0, "upper": 0.99},
            ),
        }
        if alpha_prior
        else {},
        saturation=saturation,
        saturation_params={},
        saturation_priors=saturation_params,
    )


def build_model(x, y, media, controls):

    controls_priors_sigma = 0.5
    controls_priors_mu = 0.0

    interaction = Interaction(
        formulas={"sea": "1"},
        media=media,
        controls=controls,
        is_shared_with=[],
    )

    seanson = np.array([1, -1, -1, -1]) / 2
    seansonality = 2
    beta_priors = BetaPriors(
        interaction=interaction,
        priors={},  # "beta_interaction_tv": PriorSpec("Beta", {"alpha": 3.0, "beta": 8.0})},
        media=PriorSpec("Beta", {"alpha": 3.0, "beta": 8.0}),
        control=PriorSpec(
            "Normal", {"mu": controls_priors_mu, "sigma": controls_priors_sigma}
        ),
        season=PriorSpec(
            "Normal", {"mu": seanson, "sigma": 0.1 * np.ones(seansonality * 2)}
        ),
        # season=PriorSpec("Laplace", {"mu": 0.0, "b": 0.2 * np.ones(seansonality * 2)}),
    )

    cfg = MMMConfig(
        beta_priors=beta_priors,
        media_names=media,
        control_names=controls,
        seasonality_order=seansonality,
        media_transforms={
            "tv": media_transforms(
                sigma=0.015, half_life=4.0, alpha_prior=True, saturation="Logistic"
            ),
            "crm": media_transforms(
                sigma=0.010, half_life=0.5, alpha_prior=True, saturation="Logistic"
            ),
            "social": media_transforms(
                sigma=0.010, half_life=0.5, alpha_prior=True, saturation="Logistic"
            ),
            "sea": media_transforms(
                sigma=0.010, half_life=0.5, alpha_prior=False, saturation="Logistic"
            ),
            "radio": media_transforms(
                sigma=0.010, half_life=1.0, alpha_prior=False, saturation="Logistic"
            ),
            "display": media_transforms(
                sigma=0.010, half_life=1.0, alpha_prior=False, saturation="Logistic"
            ),
            "offline": media_transforms(
                sigma=0.010, half_life=1.0, alpha_prior=False, saturation="Logistic"
            ),
            "video_&_audio": media_transforms(sigma=0.010, half_life=1.0),
        },
        random_seed=np.random.randint(0, 10000),
    )
    mmm = MMM(cfg)
    mmm.build(x, y, rescale=True)
    return mmm
