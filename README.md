# Voice Recognition & AI Assistant System

A comprehensive voice recognition and artificial intelligence system that combines audio processing, intent recognition, and machine learning models to create an intelligent conversational assistant with animal sound classification capabilities.

## 🎯 Features

### Core Capabilities
- **Voice Recording System**: High-quality audio capture with customizable duration and sample rates
- **Intent Recognition**: Natural language processing for understanding user commands and responses
- **AI Chat Assistant**: Contextual conversation system with predefined response patterns
- **Animal Sound Classification**: Machine learning models for cat and dog sound recognition
- **Multi-language Support**: Turkish language intent patterns and responses

### Technical Features
- **Neural Network Models**: Deep learning architecture using TensorFlow/Keras
- **Audio Processing**: Real-time audio streaming and WAV file generation
- **Persistent Storage**: Model serialization with pickle format
- **Early Stopping**: Intelligent training optimization to prevent overfitting
- **Embedding Layers**: Advanced text representation for natural language understanding

## 🏗️ Architecture

### Voice Processing Pipeline
1. **Audio Capture** (`ses_kaydet.py`): Records audio in configurable segments
2. **Signal Processing**: Converts audio to digital format with 16kHz sampling rate
3. **File Management**: Organizes recordings with timestamp-based naming

### AI Assistant Pipeline
1. **Intent Classification** (`model_train.py`): Processes user input through neural networks
2. **Pattern Matching**: Matches input against predefined intent patterns
3. **Response Generation**: Provides contextual responses based on detected intents
4. **Context Management**: Maintains conversation state and context

### Machine Learning Models
- **Cat/Dog Classification**: Multiple trained models for animal sound recognition
- **Text Classification**: Embedding-based neural network for intent understanding
- **Tokenizer**: Custom vocabulary management for text processing

## 📁 Project Structure

```
ses_tanima_machine2-main/
├── README.md                 # Project documentation
├── intents.json             # Intent patterns and responses
├── model_train.py           # Neural network training script
├── ses_kaydet.py            # Audio recording functionality
├── kedi_kopek_modeli*.pkl   # Pre-trained animal classification models
└── chat_model.h5            # Trained chat assistant model (generated)
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow
pip install pyaudio
pip install numpy
pip install scikit-learn
pip install wave
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ses_tanima_machine2-main.git
cd ses_tanima_machine2-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training the AI Assistant
```python
python model_train.py
```
This will:
- Load intent patterns from `intents.json`
- Train a neural network for intent classification
- Save the trained model as `chat_model.h5`
- Export tokenizer and label encoder for inference

#### Recording Audio
```python
python ses_kaydet.py
```
This will:
- Record audio in 10-second segments
- Save recordings to the `recordings/` directory
- Support extended recording sessions (up to 1 hour)

#### Supported Intents

The system currently supports the following intent categories:

- **Greeting** (`selamlama`): Hello, how are you, good morning
- **Farewell** (`veda`): Goodbye, see you later, take care
- **Entertainment** (`şaka`): Tell me a joke, make me laugh
- **Identity** (`kimlik`): Who are you, what are you
- **Programming** (`programcı`): Who made you, who programmed you
- **Environmental** (`cop_kaciyor_mu`): Are you afraid of trash

## 🧠 Model Architecture

### Neural Network Design
```
Input Layer (Text Sequence)
    ↓
Embedding Layer (16-dimensional)
    ↓
Global Average Pooling
    ↓
Dense Layer (16 units, ReLU)
    ↓
Dense Layer (16 units, ReLU)
    ↓
Output Layer (Softmax)
```

### Training Parameters
- **Vocabulary Size**: 1000 tokens
- **Maximum Sequence Length**: 20 tokens
- **Embedding Dimension**: 16
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Early Stopping**: Patience of 10 epochs

## 🔧 Configuration

### Audio Recording Settings
- **Sample Rate**: 16kHz
- **Channels**: Mono (1 channel)
- **Format**: 16-bit PCM
- **Buffer Size**: 1024 frames
- **Default Duration**: 10 seconds per segment

### Model Training Settings
- **Epochs**: Up to 200 (with early stopping)
- **Batch Size**: Default (32)
- **Validation Split**: Configurable
- **Random State**: Fixed for reproducibility

## 📊 Performance

### Model Accuracy
- **Intent Recognition**: High accuracy on training data
- **Response Time**: Real-time inference capability
- **Memory Usage**: Optimized for efficient deployment

### Audio Quality
- **Sampling Rate**: 16kHz (suitable for speech recognition)
- **Bit Depth**: 16-bit (standard for audio processing)
- **File Format**: WAV (lossless compression)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] **Multi-language Support**: English and other language intents
- [ ] **Voice Activity Detection**: Automatic speech detection
- [ ] **Real-time Streaming**: Live audio processing
- [ ] **Cloud Integration**: AWS/Azure deployment options
- [ ] **Mobile App**: Cross-platform mobile application
- [ ] **Advanced Models**: BERT/Transformer integration
- [ ] **Voice Synthesis**: Text-to-speech capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

**Muhammed Ali Taştan**
- Developing AI solutions for environmental and technological challenges
- Focus on sustainable technology and innovation

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- PyAudio community for audio processing capabilities
- Open-source contributors to the Python ecosystem



---

**Note**: This project is part of a larger initiative to develop AI assistants for environmental monitoring and cleanup operations, particularly focusing on water pollution detection and removal systems.
