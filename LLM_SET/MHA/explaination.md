# Theory

## Multi-Head Attention

*Purpose*: Multi-Head Attention (MHA) extends scaled dot-product attention by computing attention in parallel across multiple “heads,” each focusing on a different subspace of the input embeddings. This allows the model to capture diverse patterns (e.g., syntactic and semantic relationships in text).
Use Case: Core component of Transformer models (BERT, GPT, T5), used in self-attention (same sequence) and cross-attention (e.g., encoder-decoder).

*Process*:

- Projection: Linearly project $ Q, K, V $ into lower-dimensional subspaces for each head.
- Attention per Head: Apply scaled dot-product attention to each head’s $ Q_h, K_h, V_h $.
- Concatenation: Combine outputs from all heads and apply a final linear projection.
- Masking: Optionally mask positions (e.g., future tokens for autoregressive models or padding).


# Advantages:

Captures multiple aspects of the input (e.g., one head for syntax, another for semantics).
Parallelizes attention computations across heads.
Maintains expressiveness with fewer parameters per head ($ d_{\text{head}} = d_{\text{model}} / \text{num\_heads} $).





# Layman’s Explanation
Imagine you’re at a crowded party with 100 people, and you need to find the most relevant conversations to join. You’re the query ($ Q $), and each person has a key ($ K $) describing their topic and a value ($ V $) with the details of their story. You want to focus on the people whose topics match yours.

Single Attention: You compare your interest (query) to everyone’s topic (key) to calculate a “match score.” You scale these scores (to avoid shouting matches), maybe ignore some people (masking, e.g., if they’re talking about spoilers), and then weight their stories (values) based on how relevant they are. You combine these stories into one summary.
Multi-Head Attention: Instead of listening to everyone at once, you split into multiple “clones” (heads). Each clone focuses on a specific angle (e.g., one clone listens for tech talk, another for sports). Each clone computes its own match scores, weights the stories, and produces a summary. Then, all clones combine their summaries (concatenate) and polish them (final projection) into one coherent output.
Why Multiple Heads?: Different clones notice different patterns, like how tech and sports conversations relate to your interests differently. This makes your final summary richer and more nuanced.
Why Match PyTorch?: PyTorch’s MultiheadAttention is like a professional party organizer who does this perfectly. We need our custom organizer (code) to match their results exactly, ensuring we’re capturing the same conversations.
