import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Upload, 
  Image as ImageIcon, 
  Loader2, 
  AlertCircle, 
  CheckCircle, 
  RefreshCw, 
  UtensilsCrossed, 
  Info,
  Sparkles,
  Wifi,
  WifiOff,
  Play,
  Square,
  Settings,
  Activity
} from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [modelType, setModelType] = useState('custom'); // 'custom' or 'huggingface'
  
  // Backend & Model state
  const [backendStatus, setBackendStatus] = useState('checking'); // checking, connected, disconnected
  const [modelStatus, setModelStatus] = useState('demo_untrained'); // loaded, demo_untrained
  const [backendDevice, setBackendDevice] = useState('');

  // Training state
  const [trainingState, setTrainingState] = useState({
    status: 'idle',
    epochs: 3,
    current_epoch: 0,
    current_loss: 0.0,
    accuracy: 0.0,
    message: 'Model is ready to be trained.',
    error: null
  });

  const [trainConfig, setTrainConfig] = useState({
    epochs: 3,
    batch_size: 16,
    lr: 0.005,
    subset_train_size: 100,
    subset_test_size: 20
  });

  const [showConfig, setShowConfig] = useState(false);

  const fileInputRef = useRef(null);
  const pollTimerRef = useRef(null);

  // Check backend connection on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    setBackendStatus('checking');
    try {
      const response = await axios.get(`${API_BASE_URL}/`);
      if (response.data && response.data.status === 'healthy') {
        setBackendStatus('connected');
        setModelStatus(response.data.model_status);
        setBackendDevice(response.data.device || 'CPU');
      } else {
        setBackendStatus('disconnected');
      }
    } catch (err) {
      console.error("Backend health check failed:", err);
      setBackendStatus('disconnected');
    }
  };

  // Training Status Polling
  const getTrainingStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/train/status`);
      if (response.data) {
        setTrainingState(response.data);
        if (response.data.status !== 'training') {
          stopPolling();
          // Update health states to capture newly trained model if successful
          checkBackendHealth();
        }
      }
    } catch (err) {
      console.error("Error polling training status:", err);
    }
  };

  const startPolling = () => {
    if (pollTimerRef.current) return;
    pollTimerRef.current = setInterval(getTrainingStatus, 1500);
  };

  const stopPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  // Poll training status if backend is connected
  useEffect(() => {
    const checkInitialTraining = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/train/status`);
        if (response.data) {
          setTrainingState(response.data);
          if (response.data.status === 'training') {
            startPolling();
          }
        }
      } catch (err) {
        console.error("Error checking initial training status:", err);
      }
    };
    
    if (backendStatus === 'connected') {
      checkInitialTraining();
    }
    
    return () => stopPolling();
  }, [backendStatus]);

  // Handle training control
  const startTraining = async () => {
    try {
      setTrainingState(prev => ({
        ...prev,
        status: 'training',
        message: 'Mengirim perintah pelatihan...',
        error: null
      }));
      
      const response = await axios.post(`${API_BASE_URL}/train`, trainConfig);
      if (response.data) {
        startPolling();
      }
    } catch (err) {
      console.error("Error starting training:", err);
      setTrainingState(prev => ({
        ...prev,
        status: 'failed',
        error: err.response?.data?.detail || "Gagal memulai pelatihan. Silakan coba lagi.",
        message: "Gagal memulai pelatihan."
      }));
    }
  };

  const cancelTraining = async () => {
    try {
      await axios.post(`${API_BASE_URL}/train/cancel`);
      setTrainingState(prev => ({
        ...prev,
        message: 'Meminta pembatalan...'
      }));
    } catch (err) {
      console.error("Error cancelling training:", err);
    }
  };

  // Format category name (e.g. apple_pie -> Apple Pie)
  const formatFoodName = (name) => {
    if (!name) return '';
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Handle file select
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  // Process the selected file
  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (PNG, JPG, WEBP, etc.).');
      return;
    }
    
    // Clear previous results
    setError(null);
    setPrediction(null);
    setConfidence(null);
    
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Auto-submit file for prediction
    uploadAndPredict(file, modelType);
  };

  // Upload image and get prediction
  const uploadAndPredict = async (file, currentModelType = modelType) => {
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict?model_type=${currentModelType}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data) {
        setPrediction(response.data.prediction);
        setConfidence(response.data.confidence);
        setModelStatus(response.data.model_status);
      }
    } catch (err) {
      console.error("Error predicting image:", err);
      setError(
        err.response?.data?.detail || 
        "Failed to communicate with the Food Recognition server. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleModelTypeChange = (type) => {
    setModelType(type);
    if (selectedFile) {
      uploadAndPredict(selectedFile, type);
    }
  };

  // Reset form
  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPrediction(null);
    setConfidence(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top Header Navbar */}
      <header className="w-full bg-white/60 backdrop-blur-md border-b border-slate-200/80 sticky top-0 z-50 px-6 py-4 shadow-sm">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-emerald-50 rounded-xl text-white shadow-md shadow-emerald-500/20">
              <UtensilsCrossed className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-800 tracking-tight">FoodSnap AI</h1>
              <p className="text-xs text-slate-500">Image Classification & Food Recognition</p>
            </div>
          </div>

          {/* Connection Status Indicator */}
          <div className="flex items-center gap-4 text-sm">
            {backendStatus === 'checking' && (
              <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 text-slate-600 rounded-full border border-slate-200">
                <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
                <span>Checking API...</span>
              </div>
            )}
            
            {backendStatus === 'connected' && (
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 text-emerald-700 rounded-full border border-emerald-100">
                  <Wifi className="w-4 h-4 text-emerald-500" />
                  <span className="font-medium text-xs">API Connected</span>
                </div>
                
                {modelStatus === 'loaded' ? (
                  <span className="px-2.5 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-semibold border border-blue-100 flex items-center gap-1">
                    <Sparkles className="w-3.5 h-3.5 text-blue-500" /> Loaded ({backendDevice})
                  </span>
                ) : (
                  <span className="px-2.5 py-1 bg-amber-50 text-amber-700 rounded-full text-xs font-semibold border border-amber-100 flex items-center gap-1">
                    <Info className="w-3.5 h-3.5 text-amber-500" /> Demo Mode
                  </span>
                )}
              </div>
            )}

            {backendStatus === 'disconnected' && (
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 px-3 py-1.5 bg-red-50 text-red-700 rounded-full border border-red-100">
                  <WifiOff className="w-4 h-4 text-red-500" />
                  <span className="font-medium text-xs">Offline</span>
                </div>
                <button 
                  onClick={checkBackendHealth} 
                  className="p-1.5 hover:bg-slate-100 rounded-full transition-colors text-slate-500" 
                  title="Retry connection"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow max-w-6xl w-full mx-auto px-4 py-8 md:py-12 flex flex-col justify-center">
        {/* Intro Section */}
        <div className="text-center mb-8 md:mb-12">
          <span className="px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full text-xs font-medium uppercase tracking-wider">
            Powered by PyTorch ResNet18
          </span>
          <h2 className="text-3xl md:text-4xl font-extrabold text-slate-800 mt-3 tracking-tight">
            Kenali Makanan Anda dalam Sekejap
          </h2>
          <p className="text-slate-500 mt-2 max-w-xl mx-auto text-sm md:text-base">
            Unggah foto makanan Anda dan model Convolutional Neural Network (CNN) kami akan mendeteksi jenis makanan serta tingkat akurasinya.
          </p>
        </div>

        {/* 2-Column Layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
          
          {/* LEFT COLUMN: Input File Area (col-md-6 equivalent) */}
          <div className="w-full flex flex-col">
            <div className="glass-card rounded-3xl p-6 md:p-8 flex-grow flex flex-col justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-800 mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                  Unggah Gambar
                </h3>
                <p className="text-sm text-slate-500 mb-4">
                  Pilih gambar makanan dari komputer Anda. Hanya mendukung file format gambar.
                </p>
                
                {/* Model Selector Toggle */}
                <div className="mb-6 bg-slate-50/80 p-3 rounded-2xl border border-slate-100 shadow-inner">
                  <label className="text-[11px] font-bold text-slate-400 block mb-2 uppercase tracking-wider">Pilih Model Klasifikasi</label>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => handleModelTypeChange('custom')}
                      className={`flex-1 py-2 px-3 rounded-xl text-xs font-bold transition-all duration-200 ${
                        modelType === 'custom'
                          ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/10'
                          : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200/60'
                      }`}
                    >
                      Custom ResNet18 (Lokal)
                    </button>
                    <button
                      type="button"
                      onClick={() => handleModelTypeChange('huggingface')}
                      className={`flex-1 py-2 px-3 rounded-xl text-xs font-bold transition-all duration-200 ${
                        modelType === 'huggingface'
                          ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/10'
                          : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200/60'
                      }`}
                    >
                      Hugging Face ViT (Akurat)
                    </button>
                  </div>
                </div>
              </div>

              {/* Drag and Drop Zone */}
              <div 
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={triggerFileInput}
                className={`border-2 border-dashed rounded-2xl p-8 md:p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 min-h-[250px] relative ${
                  dragActive 
                    ? 'border-emerald-500 bg-emerald-50/50 scale-[0.99]' 
                    : 'border-slate-300 hover:border-emerald-400 hover:bg-slate-50/30'
                }`}
              >
                <input 
                  ref={fileInputRef}
                  type="file" 
                  accept="image/*" 
                  className="hidden" 
                  onChange={handleFileChange}
                />
                
                <div className="p-4 bg-emerald-50 text-emerald-600 rounded-2xl mb-4 group-hover:scale-110 transition-transform">
                  <Upload className="w-8 h-8" />
                </div>
                
                <p className="text-slate-700 font-semibold text-sm">
                  Klik untuk pilih file atau seret gambar ke sini
                </p>
                <p className="text-xs text-slate-400 mt-2">
                  Format yang didukung: PNG, JPG, JPEG, WEBP
                </p>

                {selectedFile && (
                  <div className="absolute bottom-4 left-4 right-4 bg-white/90 backdrop-blur-sm border border-slate-200/50 py-2 px-3 rounded-lg text-xs flex items-center justify-between text-slate-600 shadow-sm">
                    <span className="truncate font-medium max-w-[70%]">{selectedFile.name}</span>
                    <span className="text-slate-400">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="mt-6 flex items-center gap-3">
                <button
                  type="button"
                  onClick={triggerFileInput}
                  disabled={loading}
                  className="flex-grow py-3 px-4 bg-emerald-500 hover:bg-emerald-600 text-white rounded-xl text-sm font-semibold transition-all duration-200 shadow-md shadow-emerald-500/10 active:scale-95 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  <ImageIcon className="w-4 h-4" />
                  Pilih Gambar Baru
                </button>
                
                {selectedFile && (
                  <button
                    type="button"
                    onClick={handleReset}
                    className="p-3 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-xl transition-colors active:scale-95"
                    title="Reset Form"
                  >
                    <RefreshCw className="w-5 h-5" />
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Results / Preview Area (col-md-6 equivalent) */}
          <div className="w-full flex flex-col">
            <div className="glass-card rounded-3xl p-6 md:p-8 flex-grow flex flex-col justify-center items-center min-h-[350px]">
              
              {/* State 1: No Image Uploaded (Empty Div) */}
              {!previewUrl && !loading && (
                <div className="text-center p-8 flex flex-col items-center max-w-sm">
                  <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mb-4 border border-slate-200/40">
                    <UtensilsCrossed className="w-8 h-8 stroke-[1.5]" />
                  </div>
                  <h4 className="text-slate-700 font-semibold text-base mb-1">Tampilan Hasil Analisis</h4>
                  <p className="text-slate-400 text-xs leading-relaxed">
                    Gambar yang diunggah dan prediksi jenis makanannya akan ditampilkan di sini.
                  </p>
                </div>
              )}

              {/* State 2: Image Preview and Prediction */}
              {previewUrl && (
                <div className="w-full flex flex-col h-full justify-between items-center">
                  
                  {/* Image Preview Container */}
                  <div className="w-full max-h-[220px] rounded-2xl overflow-hidden shadow-md border border-slate-200 bg-slate-50 flex items-center justify-center relative group">
                    <img 
                      src={previewUrl} 
                      alt="Preview makanan" 
                      className="w-full h-full object-contain max-h-[220px] transition-transform duration-500 group-hover:scale-105"
                    />
                    
                    {loading && (
                      <div className="absolute inset-0 bg-white/70 backdrop-blur-xs flex flex-col items-center justify-center text-slate-700">
                        <Loader2 className="w-10 h-10 animate-spin text-emerald-500 mb-2" />
                        <span className="text-xs font-semibold animate-pulse">Sedang menganalisis...</span>
                      </div>
                    )}
                  </div>

                  {/* Prediction Results */}
                  <div className="w-full mt-6">
                    {error ? (
                      // Error State
                      <div className="p-4 bg-red-50 border border-red-100 text-red-700 rounded-xl text-sm flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                        <div>
                          <span className="font-semibold block">Gagal Melakukan Prediksi</span>
                          <span className="text-xs">{error}</span>
                        </div>
                      </div>
                    ) : prediction ? (
                      // Prediction State
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-xs text-slate-400 font-medium uppercase tracking-wider block">Hasil Prediksi</span>
                            <span className="text-2xl font-extrabold text-slate-800 tracking-tight">
                              {formatFoodName(prediction)}
                            </span>
                          </div>
                          <div className="text-right">
                            <span className="text-xs text-slate-400 font-medium uppercase tracking-wider block">Confidence</span>
                            <span className="text-xl font-bold text-emerald-600">
                              {(confidence * 100).toFixed(2)}%
                            </span>
                          </div>
                        </div>

                        {/* Progress Bar */}
                        <div>
                          <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-emerald-500 rounded-full transition-all duration-1000 ease-out"
                              style={{ width: `${confidence * 100}%` }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-[10px] text-slate-400 mt-1 font-medium">
                            <span>0% (Rendah)</span>
                            <span>100% (Sangat Yakin)</span>
                          </div>
                        </div>

                        {/* Status Warnings */}
                        {modelStatus === 'demo_untrained' && (
                          <div className="mt-4 p-3 bg-amber-50 border border-amber-100 text-amber-800 rounded-xl text-xs flex items-start gap-2.5">
                            <Info className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                            <p className="leading-relaxed">
                              <span className="font-bold">Perhatian:</span> Model berjalan dalam <strong>Mode Demo</strong> (bobot acak). Jalankan training di dashboard bawah atau letakkan file <code>model_cnn_project.pth</code> di folder <code>backend/</code> untuk hasil riil.
                            </p>
                          </div>
                        )}

                        {modelStatus === 'loaded' && (
                          <div className="mt-4 p-3 bg-emerald-50 border border-emerald-100 text-emerald-800 rounded-xl text-xs flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-emerald-500 shrink-0" />
                            <p className="font-medium">
                              Model kustom berhasil mendeteksi gambar secara akurat.
                            </p>
                          </div>
                        )}

                      </div>
                    ) : (
                      // Waiting for prediction state
                      !loading && (
                        <div className="text-center py-4 text-xs text-slate-400">
                          Mengirim gambar ke server untuk klasifikasi...
                        </div>
                      )
                    )}
                  </div>

                </div>
              )}

            </div>
          </div>

        </div>

        {/* Model Training Dashboard (Transfer Learning) */}
        {backendStatus === 'connected' && (
          <div className="mt-8 md:mt-12 glass-card rounded-3xl p-6 md:p-8 w-full max-w-6xl mx-auto border border-slate-200/60 shadow-lg">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-5 mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-50 text-emerald-600 rounded-xl">
                  <Activity className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-slate-800 tracking-tight">Dashboard Pelatihan Model</h3>
                  <p className="text-xs text-slate-500">Latih ulang model ResNet18 dengan Transfer Learning secara real-time</p>
                </div>
              </div>

              {/* Status Badge */}
              <div className="flex items-center gap-2 self-start md:self-auto">
                <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Status:</span>
                {trainingState.status === 'idle' && (
                  <span className="px-2.5 py-1 bg-slate-100 text-slate-700 rounded-full text-xs font-semibold border border-slate-200">
                    Ready / Idle
                  </span>
                )}
                {trainingState.status === 'training' && (
                  <span className="px-2.5 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-semibold border border-blue-100 flex items-center gap-1.5 animate-pulse">
                    <Loader2 className="w-3 h-3 animate-spin text-blue-500" /> Melatih Model
                  </span>
                )}
                {trainingState.status === 'completed' && (
                  <span className="px-2.5 py-1 bg-emerald-50 text-emerald-700 rounded-full text-xs font-semibold border border-emerald-100 flex items-center gap-1">
                    <CheckCircle className="w-3 h-3 text-emerald-500" /> Selesai (Akurat)
                  </span>
                )}
                {trainingState.status === 'cancelled' && (
                  <span className="px-2.5 py-1 bg-amber-50 text-amber-700 rounded-full text-xs font-semibold border border-amber-100">
                    Dibatalkan
                  </span>
                )}
                {trainingState.status === 'failed' && (
                  <span className="px-2.5 py-1 bg-red-50 text-red-700 rounded-full text-xs font-semibold border border-red-100 flex items-center gap-1">
                    <AlertCircle className="w-3 h-3 text-red-500" /> Gagal
                  </span>
                )}
              </div>
            </div>

            {/* Dashboard Content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Col 1: Current Progress / State */}
              <div className="lg:col-span-2 space-y-4 flex flex-col justify-between">
                <div className="space-y-4">
                  {/* Status Message */}
                  <div className="p-4 bg-slate-50 border border-slate-100 rounded-2xl">
                    <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block mb-1">Aktivitas Terkini</span>
                    <p className="text-sm text-slate-700 font-medium leading-relaxed">
                      {trainingState.message}
                    </p>
                    {trainingState.error && (
                      <p className="text-xs text-red-600 font-medium mt-2 bg-red-50/50 p-2 rounded-lg border border-red-100/50">
                        Detail: {trainingState.error}
                      </p>
                    )}
                  </div>

                  {/* Progress visualization if training */}
                  {trainingState.status === 'training' && (
                    <div className="space-y-3">
                      <div className="flex justify-between text-xs font-medium text-slate-600">
                        <span>Progress Epoch: Epoch {trainingState.current_epoch} dari {trainingState.epochs}</span>
                        <span>{Math.round((trainingState.current_epoch / trainingState.epochs) * 100)}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500 rounded-full transition-all duration-500 ease-out"
                          style={{ width: `${(trainingState.current_epoch / trainingState.epochs) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {/* Training Statistics / Metrics Grid */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-slate-50/50 border border-slate-100 p-3.5 rounded-2xl text-center">
                      <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider block">Epoch</span>
                      <span className="text-lg font-bold text-slate-700">
                        {trainingState.status === 'training' ? `${trainingState.current_epoch}/${trainingState.epochs}` : trainingState.epochs}
                      </span>
                    </div>
                    <div className="bg-slate-50/50 border border-slate-100 p-3.5 rounded-2xl text-center">
                      <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider block">Loss Terakhir</span>
                      <span className="text-lg font-bold text-slate-700">
                        {trainingState.current_loss > 0 ? trainingState.current_loss.toFixed(4) : '-'}
                      </span>
                    </div>
                    <div className="bg-slate-50/50 border border-slate-100 p-3.5 rounded-2xl text-center">
                      <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider block">Akurasi Tes</span>
                      <span className="text-lg font-bold text-emerald-600">
                        {trainingState.accuracy > 0 ? `${(trainingState.accuracy * 100).toFixed(2)}%` : '-'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Training Actions */}
                <div className="mt-6 flex flex-wrap items-center gap-3">
                  {trainingState.status === 'training' ? (
                    <button
                      type="button"
                      onClick={cancelTraining}
                      className="py-3 px-6 bg-red-500 hover:bg-red-600 text-white rounded-xl text-sm font-semibold transition-all duration-200 shadow-md shadow-red-500/10 active:scale-95 flex items-center justify-center gap-2"
                    >
                      <Square className="w-4 h-4 fill-white" />
                      Batalkan Pelatihan
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={startTraining}
                      className="py-3 px-6 bg-emerald-500 hover:bg-emerald-600 text-white rounded-xl text-sm font-semibold transition-all duration-200 shadow-md shadow-emerald-500/15 active:scale-95 flex items-center justify-center gap-2"
                    >
                      <Play className="w-4 h-4 fill-white" />
                      Mulai Pelatihan Baru
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={() => setShowConfig(!showConfig)}
                    className={`py-3 px-4 rounded-xl text-sm font-semibold transition-colors flex items-center justify-center gap-2 border ${
                      showConfig 
                        ? 'bg-slate-200 text-slate-700 border-slate-300' 
                        : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
                    }`}
                  >
                    <Settings className="w-4 h-4" />
                    Konfigurasi Parameter
                  </button>
                </div>
              </div>

              {/* Col 2: Configuration panel (either show optionally or next to it) */}
              <div className={`lg:col-span-1 border-t lg:border-t-0 lg:border-l border-slate-100 lg:pl-6 pt-6 lg:pt-0 ${showConfig ? 'block' : 'hidden lg:block opacity-60 pointer-events-none'}`}>
                <h4 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                  <Settings className="w-4 h-4 text-slate-500" />
                  Parameter Pelatihan
                </h4>
                
                <div className="space-y-4">
                  <div>
                    <label className="text-xs font-semibold text-slate-500 block mb-1">
                      Jumlah Epoch: <span className="text-slate-800">{trainConfig.epochs}</span>
                    </label>
                    <input 
                      type="range" 
                      min="1" 
                      max="10" 
                      value={trainConfig.epochs}
                      onChange={(e) => setTrainConfig(prev => ({...prev, epochs: parseInt(e.target.value)}))}
                      disabled={trainingState.status === 'training'}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500 disabled:opacity-50"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs font-semibold text-slate-500 block mb-1">Batch Size</label>
                      <select
                        value={trainConfig.batch_size}
                        onChange={(e) => setTrainConfig(prev => ({...prev, batch_size: parseInt(e.target.value)}))}
                        disabled={trainingState.status === 'training'}
                        className="w-full text-xs bg-slate-50 border border-slate-200 p-2 rounded-lg font-medium text-slate-700 focus:outline-emerald-500 disabled:opacity-50"
                      >
                        <option value={8}>8 Samples</option>
                        <option value={16}>16 Samples</option>
                        <option value={32}>32 Samples</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-slate-500 block mb-1">Learning Rate</label>
                      <select
                        value={trainConfig.lr}
                        onChange={(e) => setTrainConfig(prev => ({...prev, lr: parseFloat(e.target.value)}))}
                        disabled={trainingState.status === 'training'}
                        className="w-full text-xs bg-slate-50 border border-slate-200 p-2 rounded-lg font-medium text-slate-700 focus:outline-emerald-500 disabled:opacity-50"
                      >
                        <option value={0.01}>0.01</option>
                        <option value={0.005}>0.005</option>
                        <option value={0.001}>0.001</option>
                        <option value={0.0001}>0.0001</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-500 block mb-1">
                      Subset Data Latihan: <span className="text-slate-800">{trainConfig.subset_train_size} gambar</span>
                    </label>
                    <input 
                      type="range" 
                      min="10" 
                      max="500" 
                      step="10"
                      value={trainConfig.subset_train_size}
                      onChange={(e) => setTrainConfig(prev => ({...prev, subset_train_size: parseInt(e.target.value)}))}
                      disabled={trainingState.status === 'training'}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500 disabled:opacity-50"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-500 block mb-1">
                      Subset Data Validasi: <span className="text-slate-800">{trainConfig.subset_test_size} gambar</span>
                    </label>
                    <input 
                      type="range" 
                      min="5" 
                      max="100" 
                      step="5"
                      value={trainConfig.subset_test_size}
                      onChange={(e) => setTrainConfig(prev => ({...prev, subset_test_size: parseInt(e.target.value)}))}
                      disabled={trainingState.status === 'training'}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500 disabled:opacity-50"
                    />
                  </div>

                  <div className="p-3 bg-blue-50/50 border border-blue-100 rounded-xl">
                    <p className="text-[10px] text-blue-700 font-medium leading-relaxed flex gap-1.5 items-start">
                      <Info className="w-3.5 h-3.5 text-blue-500 shrink-0 mt-0.5" />
                      <span>
                        <strong>Tips:</strong> Jumlah data yang lebih kecil (misal: 100 train, 20 test) akan selesai dalam 1-2 menit pada CPU. Gunakan data yang lebih besar untuk mendapatkan akurasi yang lebih riil.
                      </span>
                    </p>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* Bottom troubleshooting instructions (Helpful for developer) */}
        {backendStatus === 'disconnected' && (
          <div className="mt-8 p-4 bg-slate-100 border border-slate-200 rounded-2xl max-w-2xl mx-auto">
            <h4 className="font-semibold text-slate-700 text-sm flex items-center gap-2">
              <Info className="w-4 h-4 text-slate-500" />
              Langkah-langkah untuk Menjalankan Backend:
            </h4>
            <ol className="list-decimal pl-5 text-xs text-slate-500 mt-2 space-y-1">
              <li>Buka terminal baru di direktori <code>backend/</code>.</li>
              <li>Instal dependensi jika belum: <code>pip install -r requirements.txt</code></li>
              <li>Jalankan server API dengan: <code>python app.py</code></li>
              <li>Setelah server berjalan pada <code>http://127.0.0.1:8000</code>, klik tombol refresh di pojok kanan atas halaman ini.</li>
            </ol>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full bg-slate-100 border-t border-slate-200/50 py-4 px-6 text-center text-xs text-slate-400">
        <p>© 2026 FoodSnap AI Project. Dibuat menggunakan ReactJS, Tailwind CSS, Axios, dan PyTorch.</p>
      </footer>
    </div>
  );
}

export default App;
