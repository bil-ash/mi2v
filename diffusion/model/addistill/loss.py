import torch
import torch.nn.functional as F

def gan_loss(
    noisy_real,
    noisy_fake,
    timesteps,
    y,
    model_kwargs,

    gan_loss_type,
    disc_backbone,
    discriminator,

    # down_intrablock_additional_residuals=None,
    step=0,
):

    # # Sample noise
    # noise = torch.randn_like(student_output)

    # # Selected timesteps
    # selected_timesteps = [10, 250, 500, 750]
    # prob = torch.tensor([0.25, 0.25, 0.25, 0.25])

    # # Sample the timesteps
    # idx = prob.multinomial(student_output.shape[0], replacement=True).to(student_output.device)
    # timesteps = torch.tensor(selected_timesteps, device=student_output.device, dtype=torch.long)[idx]

    # # Create noisy sample
    # noisy_fake = teacher_noise_scheduler.add_noise(student_output, noise, timesteps)
    # noisy_real = teacher_noise_scheduler.add_noise(real, noise, timesteps)

    # # Concatenate noisy samples
    # noisy_sample = torch.cat([noisy_fake, noisy_real], dim=0)

    # # Concatenate conditionings
    # if conditioning is not None:
    #     conditioning = {
    #         "cond": {k: torch.cat([v, v], dim=0) for k, v in conditioning["cond"].items()}
    #     }

    # # Concatenate timesteps
    # timestep = torch.cat([timesteps, timesteps], dim=0)

    # if down_intrablock_additional_residuals is not None:
    #     for k, v in enumerate(down_intrablock_additional_residuals):
    #         down_intrablock_additional_residuals[k] = torch.cat([v, v], dim=0)
    # else:
    #     down_intrablock_additional_residuals = None

    # # Predict noise level using denoiser
    # denoised_sample = disc_backbone(
    #     sample=noisy_sample,
    #     timestep=timestep,
    #     conditioning=conditioning,
    #     down_intrablock_additional_residuals=down_intrablock_additional_residuals,
    #     return_intermediate=True,
    # )

    # denoised_sample_fake, denoised_sample_real = denoised_sample.chunk(2, dim=0)
    denoised_sample_fake = disc_backbone(noisy_fake, timesteps, guide_image=None, y=y, cond_mask=None, flow_score=None, **model_kwargs)
    denoised_sample_real = disc_backbone(noisy_real, timesteps, guide_image=None, y=y, cond_mask=None, flow_score=None, **model_kwargs)

    if gan_loss_type == "wgan":
        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
        if step % 2 == 0:
            loss_G = -discriminator(denoised_sample_fake).mean()
            loss_D = 0
        else:
            loss_D = (
                -discriminator(denoised_sample_real).mean()
                + discriminator(denoised_sample_fake.detach()).mean()
            )
            loss_G = 0

    elif gan_loss_type == "lsgan":
        valid = torch.ones(noisy_fake.size(0), 1, device=noisy_fake.device)
        fake = torch.zeros(noisy_fake.size(0), 1, device=noisy_fake.device)
        if step % 2 == 0:
            loss_G = F.mse_loss(
                torch.sigmoid(discriminator(denoised_sample_fake)), valid
            )
            # loss_D = 0
            loss_D = torch.tensor(0.0)
        else:
            loss_D = 0.5 * (
                F.mse_loss(
                    torch.sigmoid(discriminator(denoised_sample_real)), valid
                )
                + F.mse_loss(
                    torch.sigmoid(discriminator(denoised_sample_fake.detach())),
                    fake,
                )
            )
            # loss_G = 0
            loss_G = torch.tensor(0.0)
    elif gan_loss_type == "hinge":
        if step % 2 == 0:
            loss_G = -discriminator(denoised_sample_fake).mean()
            # loss_D = 0
            loss_D = torch.tensor(0.0)
        else:
            loss_D = (
                F.relu(1.0 - discriminator(denoised_sample_real)).mean()
                + F.relu(1.0 + discriminator(denoised_sample_fake.detach())).mean()
            )
            # loss_G = 0
            loss_G = torch.tensor(0.0)

    elif gan_loss_type == "non-saturating":
        if step % 2 == 0:
            loss_G = -torch.mean(
                torch.log(torch.sigmoid(discriminator(denoised_sample_fake)) + 1e-8)
            )
            loss_D = 0
        else:
            loss_D = -torch.mean(
                torch.log(torch.sigmoid(discriminator(denoised_sample_real)) + 1e-8)
                + torch.log(
                    1
                    - torch.sigmoid(discriminator(denoised_sample_fake.detach()))
                    + 1e-8
                )
            )
            loss_G = 0
    else:
        if step % 2 == 0:
            valid = torch.ones(noisy_fake.size(0), 1, device=noisy_fake.device)
            loss_G = F.binary_cross_entropy_with_logits(
                discriminator(denoised_sample_fake), valid
            )
            loss_D = 0
        else:
            valid = torch.ones(noisy_fake.size(0), 1, device=noisy_fake.device)
            real = F.binary_cross_entropy_with_logits(
                discriminator(denoised_sample_real), valid
            )
            fake = torch.zeros(noisy_fake.size(0), 1, device=noisy_fake.device)
            fake = F.binary_cross_entropy_with_logits(
                discriminator(denoised_sample_fake.detach()), fake
            )
            loss_D = real + fake
            loss_G = 0

    return [loss_G, loss_D]
