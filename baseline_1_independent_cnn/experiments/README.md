# Baseline 1 — Independent CNN experiments

These folders contain the original notebooks, saved results, figures, configurations and diagnostic exports; their internal structure and file bytes are unchanged.

| Folder | Contents |
| --- | --- |
| [efficientnet_b0](efficientnet_b0/) | ImageNet-pretrained EfficientNet-B0 training, dropout, weight decay and staged evaluation |
| [resnet50](resnet50/) | ResNet-50 freezing, fine-tuning, disease and generalisation experiments |
| [convnext_tiny](convnext_tiny/) | ConvNeXt-Tiny training and evaluation |
| [consolidated_results](consolidated_results/) | Original AML_B1_Clean_Results folder, with consolidated development, fine-tuning, official-test and selection records |

**Important:** The `consolidated_results` folder is preserved whole, not merged or deduplicated against other experiments. For the documented baseline selection see [../final](../final/). Development and official-test results should be reported separately.
