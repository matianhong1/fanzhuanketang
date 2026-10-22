# Shared project contract

Root: F:/jiqixuexiPythonshijian. Python package traffic_signs. CLI main.py owned by root.
All paths in config relative to project root. UTF-8 files. No PPT.

Ownership: Agent A data.py + labels.py + scripts/download_data.py. Agent B features.py. Agent C models.py. Root config, CLI, evaluation, plots, reporting, tests, README and integration.

Config dict: seed=42, classes=list(range(43)), image_size=32, use_roi=false (new input already cropped; optional ROI requires explicit metadata), validation_fraction=0.2, hog={orientations:9,pixels_per_cell:[8,8],cells_per_block:[2,2],block_norm:'L2-Hys',transform_sqrt:true}, data_dir='data', output_dir='outputs/full', assets_dir='presentation_assets', robust_per_class=30, svm_C=[0.01,0.1,1.0], knn_k=[3,5].

Manifest pandas DataFrame fields: path (absolute), label (original int ID), split (train/val/test), track (classID_trackID for training, test_filename for test), x1,y1,x2,y2 ROI inclusive coordinates, width,height. No augmentation. All splits saved. Data API prepare_data(config)->DataFrame with checks and outputs manifest.csv, data_audit.json, class_counts.csv under output_dir; download script official ZIPs into data/raw then extracted to data/extracted. labels.py CLASS_NAMES dict[int,str].

Features API: preprocess(path, config, roi=None, brightness=1.0, blur=0.0)->2D float32 gray in [0,1]. load_rgb(path,config,roi=None,brightness=1.0,blur=0.0)->PIL RGB resized (or expose equivalent). extract_features(manifest, config, kind='hog', brightness=1.0, blur=0.0, cache=True)->np.ndarray aligned manifest rows. feature_vector(gray,config,kind='hog')->1D float32. hog_visualization(gray,config)->2D array. Cache under output_dir/cache keyed by input metadata, manifest order and complete preprocessing/HOG configuration. Distort before resize/grayscale, darken then blur. Same API for prediction. B owns optional feature visual plot helpers only within features.py.

Models API train_models(manifest, config)->dict name:bundle. Names hog_svm,hog_knn,pixels_svm. Bundle dict pipeline (fitted sklearn Pipeline), kind ('hog'/'pixels'), config, classes, name, selected_params, timings {train_seconds,...}. Parameter selection on train/val only by macro F1, then final refit train+val before test. Save output_dir/models/{name}.joblib and validation_search.csv; model_summary.json. Scalar timing must distinguish tuning/refit/features and prediction. All three models same data, scaler fit only on appropriate training pool, same SVM C candidates. LinearSVC dual=False max_iter=10000 recommended for CPU. Saved bundle includes full config and Chinese names; root implements raw image prediction using features API. C may coordinate API adjustments via messages.

Root evaluation does held-out tests, raw metrics/predictions/confusions, robustness stratified fixed test subset, saved/reloaded consistency, plots and material index. Run quick classes 0..9 then full 0..42. Agent A should actually download/prepare official data; B/C may wait for it but own modules can be validated independently. Do not modify other owners' files.
