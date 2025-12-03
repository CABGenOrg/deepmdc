import torch
from omegaconf import DictConfig
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, \
    SequenceEmbeddingLSTM, SequenceEmbeddingCNNEnhanced, AttentionNetwork, \
    OutputNetwork


def build_model(cfg: DictConfig) -> DeepRC:
    """
    Constructs and returns a DeepRC model based on the given configuration.

    Args:
        cfg (DictConfig): Configuration object containing model parameters,
        including CNN, attention, and output network settings.

    Returns:
        DeepRC: A DeepRC model instance with the specified architecture and
        parameters.
    """

    # Select the device (GPU if available, otherwise CPU)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Sequence embedding configuration

    # Sequence embedding type
    sequence_embedding_type = cfg.model.sequence_embedding.type
    # Size of convolutional kernels
    kernel_size = cfg.model.sequence_embedding.kernel_size
    # Number of convolutional filters or LSTM blocks
    sequence_embedding_n_units = cfg.model.sequence_embedding.n_units
    # Number of sequence embedding network
    sequence_embedding_layers = cfg.model.sequence_embedding.n_layers
    # Sizes of convolutional kernels
    kernel_sizes = cfg.model.sequence_embedding.kernel_sizes
    # Number of convolutional blocks per scale
    n_conv_blocks = cfg.model.sequence_embedding.n_conv_blocks
    # Whether the information should be sent directly to the posterior layers
    use_residual = cfg.model.sequence_embedding.use_residual
    # Whether to find the most important kernel for each sequence
    use_squeeze_excitation = \
        cfg.model.sequence_embedding.use_squeeze_excitation
    # Whether to use depthwise separable convolutions instead of standard
    # convolutions
    use_depthwise_separable = \
        cfg.model.sequence_embedding.use_depthwise_separable
    # Global pooling method
    pooling_method = cfg.model.sequence_embedding.pooling_method

    # Attention mechanism configuration

    # Number of attention layers
    attention_layers = cfg.model.attention.n_layers
    # Number of attention units per layer
    attention_units = cfg.model.attention.n_units

    # Output network configuration
    # Number of fully connected layers
    output_layers = cfg.model.output.n_layers
    # Number of units in each output layer
    output_units = cfg.model.output.n_units

    # Determine the number of output features based on the type of
    # classification task
    multiclass_targets = cfg.task.target.get(
        "possible_target_values")  # Multi-class labels
    # Binary classification target
    binary_targets = cfg.task.target.positive_class

    # Number of output features depends on the classification type
    n_output_features = len(multiclass_targets) if multiclass_targets is not \
        None else len(binary_targets)

    print(f"Using device: {device}")
    print("Reconstructing the model architecture...")

    # Define the sequence embedding network
    sequence_embedding_network = None
    if sequence_embedding_type == "CNN":
        sequence_embedding_network = SequenceEmbeddingCNN(
            # 20 standard amino acid features + 3 additional features
            n_input_features=20+3,
            kernel_size=kernel_size,
            n_kernels=sequence_embedding_n_units,
            n_layers=sequence_embedding_layers
        )
    elif sequence_embedding_type == "LSTM":
        sequence_embedding_network = SequenceEmbeddingLSTM(
            # 20 standard amino acid features + 3 additional features
            n_input_features=20+3,
            n_lstm_blocks=sequence_embedding_n_units,
            n_layers=sequence_embedding_layers
        )
    elif sequence_embedding_type == "CNNEnhanced":
        sequence_embedding_network = SequenceEmbeddingCNNEnhanced(
            n_input_featues=20+3,
            n_kernels=sequence_embedding_n_units,
            kernel_sizes=kernel_sizes,
            n_conv_blocks=n_conv_blocks,
            use_residual=use_residual,
            use_squeeze_excitation=use_squeeze_excitation,
            use_depthwise_separable=use_depthwise_separable,
            pooling_method=pooling_method
        )
    else:
        raise ValueError("Invalid sequence embedding type: "
                         f"{sequence_embedding_type}. Available only CNN, LSTM"
                         " and CNNEnhanced.")

    embedding_features = sequence_embedding_network.output_dim \
        if isinstance(sequence_embedding_network,
                      SequenceEmbeddingCNNEnhanced) \
        else sequence_embedding_n_units

    # Define the attention network for sequence feature aggregation
    attention_network = AttentionNetwork(
        # Output from Sequence embedding
        n_input_features=embedding_features,
        n_layers=attention_layers,
        n_units=attention_units,
    )

    # Define the output network responsible for final classification
    output_network = OutputNetwork(
        # Output from attention mechanism
        n_input_features=embedding_features,
        n_output_features=n_output_features,  # Number of classes
        n_layers=output_layers,
        n_units=output_units
    )

    # Construct the DeepRC model with the defined components
    model = DeepRC(
        # Maximum sequence length supported by the model
        max_seq_len=cfg.max_seq_len,
        sequence_embedding_network=sequence_embedding_network,
        attention_network=attention_network,
        output_network=output_network,
        consider_seq_counts=False,  # Ignore sequence count information
        n_input_features=20,  # Number of input features per residue
        add_positional_information=True,  # Include positional encodings
        # Reduce sequence length by 10% during processing
        sequence_reduction_fraction=0.1,
        # Batch size for memory-efficient processing
        reduction_mb_size=int(5e4),
        device=device  # Device to run the model on (CPU/GPU)
    ).to(device)

    return model


def clear_gpu_memory():
    """Release unused VRAM before and after training."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
