def apply():
    import comfy.clip_model
    import comfy.text_encoders.t5

    def clip_layer_forward(self, x, mask=None, optimized_attention=None):
        x = x + self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x = x + self.mlp(self.layer_norm2(x))
        return x

    def t5_ff_forward(self, x):
        forwarded_states = self.DenseReluDense(self.layer_norm(x))
        return x + forwarded_states

    def t5_attention_forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        output, past_bias = self.SelfAttention(
            self.layer_norm(x),
            mask=mask,
            past_bias=past_bias,
            optimized_attention=optimized_attention,
        )
        return x + output, past_bias

    comfy.clip_model.CLIPLayer.forward = clip_layer_forward
    comfy.text_encoders.t5.T5LayerFF.forward = t5_ff_forward
    comfy.text_encoders.t5.T5LayerSelfAttention.forward = t5_attention_forward
