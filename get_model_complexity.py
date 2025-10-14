import torch
from ecg_models import ECGSMARTNET, ECGSMARTNET_Attention


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_flops(model: torch.nn.Module, x: torch.Tensor) -> int | None:
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(x,), verbose=False)
    # FLOPs ~ 2 * MACs (multiply-adds)
    return int(2 * macs)


def fmt_big(n: int, unit: str) -> str:
    if unit == "params":
        return f"{n/1e6:.2f}M"
    if unit == "flops":
        return f"{n/1e9:.3f}G"
    return str(n)


def report(name: str, model: torch.nn.Module, x: torch.Tensor) -> None:
    params = count_params(model)
    flops = measure_flops(model, x)
    if flops is None:
        print(f"{name}: params={fmt_big(params,'params')} | FLOPs: install 'thop' to compute")
    else:
        print(f"{name}: params={fmt_big(params,'params')} | FLOPs={fmt_big(flops,'flops')}")


if __name__ == "__main__":
    # Dummy input: (batch, channels, leads, time)
    x = torch.randn(1, 1, 12, 200)

    models = [
        ("ECGSMARTNET", ECGSMARTNET()),
        ("ECGSMARTNET+SE", ECGSMARTNET_Attention(attention='se')),
        ("ECGSMARTNET+CBAM", ECGSMARTNET_Attention(attention='cbam')),
    ]

    for name, model in models:
        report(name, model, x)


