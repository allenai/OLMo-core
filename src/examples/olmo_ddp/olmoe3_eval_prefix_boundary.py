"""One-GPU regression for the actual medium eval expert-row address boundary."""

import torch

from olmo_core.kernels.swiglu import swiglu_backward_valid_prefix, swiglu_valid_prefix


def main():
    """Check every output in bounded reference chunks, including observed718369 rows."""
    torch.cuda.set_device(0)
    for rows in (1024, 699_040, 718_369):
        print("EVAL_PREFIX_BOUNDARY_START", rows, rows * 3072, flush=True)
        x = torch.empty((rows, 3072), device="cuda", dtype=torch.bfloat16)
        x[:, :1536].fill_(0.25)
        x[:, 1536:].fill_(1.5)
        valid = torch.tensor(rows, device="cuda", dtype=torch.int64)
        with torch.no_grad():
            out = swiglu_valid_prefix(x, valid)
        torch.cuda.synchronize()
        value = (0.25 * 1.5 * torch.sigmoid(torch.tensor(1.5))).to(torch.bfloat16)
        for chunk in out.split(4096):
            torch.testing.assert_close(chunk, torch.full_like(chunk, value.item()), rtol=0, atol=0)
        print("EVAL_PREFIX_BOUNDARY_PASS", rows, flush=True)
        del out
        grad_h = torch.full((rows, 1536), 0.75, device="cuda", dtype=torch.bfloat16)
        dx = swiglu_backward_valid_prefix(x, grad_h, valid)
        torch.cuda.synchronize()
        sig = torch.sigmoid(torch.tensor(1.5))
        expected_up = (0.75 * 1.5 * sig).to(torch.bfloat16).item()
        expected_gate = (0.75 * 0.25 * sig * (1 + 1.5 * (1 - sig))).to(torch.bfloat16).item()
        for chunk in dx.split(4096):
            torch.testing.assert_close(
                chunk[:, :1536], torch.full_like(chunk[:, :1536], expected_up), rtol=0, atol=0
            )
            torch.testing.assert_close(
                chunk[:, 1536:], torch.full_like(chunk[:, 1536:], expected_gate), rtol=0, atol=0
            )
        print("EVAL_PREFIX_BACKWARD_BOUNDARY_PASS", rows, flush=True)
        del x, dx, grad_h, valid
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
