
// Define colors for different block categories
const MODEL_BLOCK_COLOR = 230;
const LAYER_BLOCK_COLOR = 210;
const DATA_BLOCK_COLOR = 160;

// --- Model Definition Blocks ---

Blockly.Blocks['model_builder'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Sequential Model");
    this.appendStatementInput("LAYERS")
        .setCheck("Layer")
        .appendField("Layers");
    this.appendStatementInput("COMPILER")
        .setCheck("Compiler")
        .appendField("Compile Settings");
    this.setColour(MODEL_BLOCK_COLOR);
    this.setTooltip('Defines a sequential neural network model.');
    this.setHelpUrl('');
  }
};

Blockly.Blocks['model_compiler'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Compile Model");
    this.appendValueInput("OPTIMIZER")
        .setCheck("Optimizer")
        .setAlign(Blockly.ALIGN_RIGHT)
        .appendField("Optimizer");
    this.appendDummyInput()
        .setAlign(Blockly.ALIGN_RIGHT)
        .appendField("Loss Function")
        .appendField(new Blockly.FieldDropdown([
            ["Categorical Cross-Entropy", "categoricalCrossentropy"],
            ["Mean Squared Error", "meanSquaredError"],
            ["Binary Cross-Entropy", "binaryCrossentropy"]
        ]), "LOSS");
    this.setPreviousStatement(true, 'Compiler');
    this.setColour(MODEL_BLOCK_COLOR);
    this.setTooltip('Configures the model for training.');
    this.setHelpUrl('');
  }
};

// --- Layer Blocks ---

Blockly.Blocks['layer_input'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Input Layer");
    this.appendDummyInput()
        .appendField("  Image Size:")
        .appendField(new Blockly.FieldNumber(64, 1), "WIDTH")
        .appendField("x")
        .appendField(new Blockly.FieldNumber(64, 1), "HEIGHT")
        .appendField("px");
    this.appendDummyInput()
        .appendField("  Channels:")
        .appendField(new Blockly.FieldDropdown([["RGB (3)", "3"], ["Grayscale (1)", "1"]]), "CHANNELS");
    this.setPreviousStatement(true, "Layer");
    this.setNextStatement(true, "Layer");
    this.setColour(LAYER_BLOCK_COLOR);
    this.setTooltip('Defines the input shape for the model.');
    this.setHelpUrl('');
  }
};

Blockly.Blocks['layer_dense'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Dense Layer (Fully Connected)");
    this.appendDummyInput()
        .setAlign(Blockly.ALIGN_RIGHT)
        .appendField("Units")
        .appendField(new Blockly.FieldNumber(128, 1), "UNITS");
    this.appendDummyInput()
        .setAlign(Blockly.ALIGN_RIGHT)
        .appendField("Activation")
        .appendField(new Blockly.FieldDropdown([
            ["ReLU", "relu"],
            ["Sigmoid", "sigmoid"],
            ["Softmax", "softmax"],
            ["Linear", "linear"]
        ]), "ACTIVATION");
    this.setPreviousStatement(true, "Layer");
    this.setNextStatement(true, "Layer");
    this.setColour(LAYER_BLOCK_COLOR);
    this.setTooltip('A standard fully connected layer.');
    this.setHelpUrl('');
  }
};

Blockly.Blocks['layer_conv2d'] = {
    init: function() {
        this.appendDummyInput()
            .appendField("Conv2D Layer");
        this.appendDummyInput()
            .setAlign(Blockly.ALIGN_RIGHT)
            .appendField("Filters")
            .appendField(new Blockly.FieldNumber(32, 1), "FILTERS");
        this.appendDummyInput()
            .setAlign(Blockly.ALIGN_RIGHT)
            .appendField("Kernel Size")
            .appendField(new Blockly.FieldNumber(3, 1), "KERNEL_SIZE");
        this.appendDummyInput()
            .setAlign(Blockly.ALIGN_RIGHT)
            .appendField("Activation")
            .appendField(new Blockly.FieldDropdown([
                ["ReLU", "relu"],
                ["Sigmoid", "sigmoid"],
                ["Linear", "linear"]
            ]), "ACTIVATION");
        this.setPreviousStatement(true, "Layer");
        this.setNextStatement(true, "Layer");
        this.setColour(LAYER_BLOCK_COLOR);
        this.setTooltip('2D convolution layer for feature extraction from images.');
        this.setHelpUrl('');
    }
};

Blockly.Blocks['layer_maxpooling2d'] = {
    init: function() {
        this.appendDummyInput()
            .appendField("MaxPooling2D Layer");
        this.appendDummyInput()
            .setAlign(Blockly.ALIGN_RIGHT)
            .appendField("Pool Size")
            .appendField(new Blockly.FieldNumber(2, 1), "POOL_SIZE");
        this.setPreviousStatement(true, "Layer");
        this.setNextStatement(true, "Layer");
        this.setColour(LAYER_BLOCK_COLOR);
        this.setTooltip('Down-samples the input along its spatial dimensions.');
        this.setHelpUrl('');
    }
};

Blockly.Blocks['layer_flatten'] = {
    init: function() {
        this.appendDummyInput()
            .appendField("Flatten Layer");
        this.setPreviousStatement(true, "Layer");
        this.setNextStatement(true, "Layer");
        this.setColour(LAYER_BLOCK_COLOR);
        this.setTooltip('Flattens the input, preparing it for Dense layers.');
        this.setHelpUrl('');
    }
};

Blockly.Blocks['layer_dropout'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Dropout Layer");
    this.appendDummyInput()
        .setAlign(Blockly.ALIGN_RIGHT)
        .appendField("Rate")
        .appendField(new Blockly.FieldNumber(0.2, 0, 1, 0.01), "RATE");
    this.setPreviousStatement(true, "Layer");
    this.setNextStatement(true, "Layer");
    this.setColour(LAYER_BLOCK_COLOR);
    this.setTooltip('Applies dropout to the input, helping prevent overfitting.');
    this.setHelpUrl('');
  }
};

// --- Optimizer Blocks ---

Blockly.Blocks['optimizer_adam'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Adam Optimizer");
    this.appendDummyInput()
        .setAlign(Blockly.ALIGN_RIGHT)
        .appendField("Learning Rate")
        .appendField(new Blockly.FieldNumber(0.001, 0), "LEARNING_RATE");
    this.setOutput(true, "Optimizer");
    this.setColour(MODEL_BLOCK_COLOR - 40);
    this.setTooltip('Adam optimizer.');
    this.setHelpUrl('');
  }
};

// --- JavaScript Generators ---

// The following logic is not standard Blockly JavaScript generation,
// but is used to create a JSON structure for our application.

Blockly.JavaScript['model_builder'] = function(block) {
    const layersCode = Blockly.JavaScript.statementToCode(block, 'LAYERS').trim();
    const finalLayersCode = layersCode.endsWith(',') ? layersCode.slice(0, -1) : layersCode;
    const code = `{
  "layers": [
${finalLayersCode}
  ]
}`;
    return [code, Blockly.JavaScript.ORDER_ATOMIC];
};

Blockly.JavaScript['model_compiler'] = function(block) {
    const optimizer = Blockly.JavaScript.valueToCode(block, 'OPTIMIZER', Blockly.JavaScript.ORDER_ATOMIC) || 'null';
    const loss = block.getFieldValue('LOSS');
    const code = `{
  "optimizer": ${optimizer.trim()},
  "loss": "${loss}"
}`;
    return [code, Blockly.JavaScript.ORDER_ATOMIC];
};

Blockly.JavaScript['layer_input'] = function(block) {
    const width = block.getFieldValue('WIDTH');
    const height = block.getFieldValue('HEIGHT');
    const channels = block.getFieldValue('CHANNELS');
    const code = `    {
      "type": "input",
      "shape": [${height}, ${width}, ${channels}]
    },`;
    return code;
};

Blockly.JavaScript['layer_dense'] = function(block) {
    const units = block.getFieldValue('UNITS');
    const activation = block.getFieldValue('ACTIVATION');
    const code = `    {
      "type": "dense",
      "units": ${units},
      "activation": "${activation}"
    },`;
    return code;
};

Blockly.JavaScript['layer_conv2d'] = function(block) {
    const filters = block.getFieldValue('FILTERS');
    const kernelSize = block.getFieldValue('KERNEL_SIZE');
    const activation = block.getFieldValue('ACTIVATION');
    const code = `    {
      "type": "conv2d",
      "filters": ${filters},
      "kernelSize": ${kernelSize},
      "activation": "${activation}"
    },`;
    return code;
};

Blockly.JavaScript['layer_maxpooling2d'] = function(block) {
    const poolSize = block.getFieldValue('POOL_SIZE');
    const code = `    {
      "type": "maxPooling2d",
      "poolSize": ${poolSize}
    },`;
    return code;
};

Blockly.JavaScript['layer_flatten'] = function(block) {
    const code = `    {
      "type": "flatten"
    },`;
    return code;
};

Blockly.JavaScript['layer_dropout'] = function(block) {
    const rate = block.getFieldValue('RATE');
    const code = `    {
      "type": "dropout",
      "rate": ${rate}
    },`;
    return code;
};

Blockly.JavaScript['optimizer_adam'] = function(block) {
    const learningRate = block.getFieldValue('LEARNING_RATE');
    const code = `{  "type": "adam",  "learningRate": ${learningRate}}`;
    return [code, Blockly.JavaScript.ORDER_ATOMIC];
};