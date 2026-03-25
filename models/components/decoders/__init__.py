from .mlp_decoder import MLPDecoder
from .structured_moflow_decoder import StructuredMoFlowDecoder


def build_sequence_decoder(name, model_cfg, latent_dim, ctx_dim, out_dim, num_agents):
    if name == "mlp":
        hidden_dim = model_cfg.MOTION_DECODER.D_MODEL
        return MLPDecoder(
            latent_dim=latent_dim,
            ctx_dim=ctx_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
        )
    if name == "moflow_structured":
        return StructuredMoFlowDecoder(
            model_cfg=model_cfg,
            latent_dim=latent_dim,
            ctx_dim=ctx_dim,
            out_dim=out_dim,
            num_agents=num_agents,
        )
    raise ValueError(f"Unknown decoder '{name}'")


__all__ = [
    "MLPDecoder",
    "StructuredMoFlowDecoder",
    "build_sequence_decoder",
]
