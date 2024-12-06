
Here an example on how to use the yaml files
```python
import yaml

# Load the configuration file
config_path = 'path/to/cnn_config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
```

```python
# Update the input_shape and output_shape in the config
config['data']['input_shape'] = [input_shape[0], 1]
config['data']['output_shape'] = output_shape

# Build the model
models_dir = os.path.join(code_dir, 'models')
sys.path.append(models_dir)
from CNN import build_CNN

model = build_CNN(
    input_shape=tuple(config['data']['input_shape']),
    output_shape=config['data']['output_shape'],
    layer_1_size=config['model']['layer_1_size'],
    layer_2_size=config['model']['layer_2_size'],
    layer_3_size=config['model']['layer_3_size'],
    layer_FC_size=config['model']['layer_FC_size'],
    dropout_rate=config['model']['dropout_rate']
)
```