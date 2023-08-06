import argparse
import torch
from builder import export_from_registry, model_registry
from core.utils.visualize import show_supported_models_on_command_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    opts = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg, _, trainer = export_from_registry(opts.model)

    trainer(model_cfg, device).train()


if __name__ == '__main__':
    main()
