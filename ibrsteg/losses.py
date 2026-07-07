from __future__ import annotations


def attribute_loss(attribute_losses, loss_cfg):
    """Weighted 2D Gaussian Attribute Map loss from the paper."""

    return loss_cfg.lambda2d * (
        loss_cfg.lambdaemb
        * (weighted_attribute_loss(attribute_losses[0], loss_cfg) + weighted_attribute_loss(attribute_losses[1], loss_cfg))
        + loss_cfg.lambdare
        * (weighted_attribute_loss(attribute_losses[2], loss_cfg) + weighted_attribute_loss(attribute_losses[3], loss_cfg))
    )


def weighted_attribute_loss(loss_tuple, loss_cfg):
    depth_loss, rgb_loss, opacity_loss = loss_tuple
    return (
        depth_loss * loss_cfg.lambdadepth
        + rgb_loss * loss_cfg.lambdargb
        + opacity_loss * loss_cfg.lambdaalpha
    )
