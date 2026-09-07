"""One-GPU regression for the actual medium eval expert-row address boundary."""

import torch

from olmo_core.kernels.swiglu import swiglu_valid_prefix


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
        del x, out, valid
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
