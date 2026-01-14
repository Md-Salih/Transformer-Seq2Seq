# Transformer Seq2Seq - React + Vite Frontend

Modern web application for text summarization powered by Transformer architecture.

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm/yarn
- Expo CLI: `npm install -g expo-cli`
- For mobile: Expo Go app on your device

### Installation

```bash
cd frontend
npm install
```

### Run the App

**Web:**
```bash
npm run web
```

**iOS Simulator:**
```bash
npm run ios
```

**Android Emulator:**
```bash
npm run android
```

**On Physical Device:**
```bash
npm start
# Scan QR code with Expo Go app
```

## 🏗️ Architecture

### Frontend (React Native)
- **Framework:** Expo (React Native)
- **Cross-platform:** iOS, Android, Web (React Native Web)
- **Animations:** React Native Animated API
- **State Management:** React Hooks

### Backend Integration
- Connects to Flask API at `http://localhost:5000`
- Server-Sent Events (SSE) for streaming token generation
- Real-time progress updates and animations

## 📱 Features

### Universal Platform Support
- **Mobile Apps:** Native iOS and Android apps
- **Web App:** Responsive web application
- **Single Codebase:** Write once, run everywhere

### Professional UI
- Dark theme inspired by modern dashboards
- Cyan accent color (#00d4aa) on navy background (#0a0e1a)
- Smooth animations for token-by-token generation
- Responsive design for all screen sizes

### Core Functionality
- Real-time text summarization
- Token-by-token streaming display
- Processing statistics (compression ratio, time)
- Quick example documents
- Architecture visualization

## 🎨 Design System

### Colors
- **Background:** #0a0e1a (Dark Navy)
- **Card Background:** #0f172a
- **Accent:** #00d4aa (Cyan)
- **Text Primary:** #f1f5f9
- **Text Secondary:** #94a3b8
- **Borders:** #1e293b, #334155

### Typography
- System fonts for native look on each platform
- Font sizes: 13-24px range
- Weight: 400 (regular), 600 (semibold), 700 (bold)

## 🛠️ Development

### Project Structure
```
frontend/
├── App.js              # Main application component
├── app.json            # Expo configuration
├── package.json        # Dependencies
├── babel.config.js     # Babel configuration
└── index.js           # Entry point
```

### Backend Setup
Ensure the Flask backend is running:
```bash
cd ..
python app.py
```

### Building for Production

**Web:**
```bash
npm run build:web
# Output in web-build/ directory
```

**iOS:**
```bash
expo build:ios
```

**Android:**
```bash
expo build:android
```

## 🔧 Configuration

### API Endpoint
Update `API_URL` in `App.js`:
- **Web:** `http://localhost:5000`
- **Android Emulator:** `http://10.0.2.2:5000`
- **iOS Simulator/Physical Device:** Use your computer's IP address

### Customization
- **Colors:** Modify `styles` object in `App.js`
- **Examples:** Update `EXAMPLES` array
- **Animations:** Adjust `Animated.timing` duration values

## 📦 Dependencies

### Core
- `expo` - Development platform
- `react-native` - Mobile framework
- `react-native-web` - Web support

### UI/Animations
- `react-native-reanimated` - High-performance animations
- `expo-linear-gradient` - Gradient support
- `react-native-svg` - SVG rendering

## 🌐 Deployment

### Web Deployment
Deploy the `web-build/` folder to any static hosting:
- Netlify
- Vercel
- GitHub Pages
- AWS S3 + CloudFront

### Mobile App Stores
- **iOS:** Submit to App Store via Expo EAS Build
- **Android:** Submit to Google Play via Expo EAS Build

## 📝 Notes

- **Network Requests:** Uses Fetch API with streaming support
- **Animations:** Optimized with `useNativeDriver` where possible
- **Performance:** Efficient re-renders with React hooks
- **Accessibility:** Follows React Native accessibility guidelines

## 🐛 Troubleshooting

**CORS Issues (Web):**
Add CORS headers to Flask backend:
```python
from flask_cors import CORS
CORS(app)
```

**Android Network Error:**
Ensure using `10.0.2.2` for Android emulator localhost

**Metro Bundler Issues:**
```bash
npx expo start --clear
```

## 📄 License

MIT License - Production-ready Transformer Seq2Seq application
