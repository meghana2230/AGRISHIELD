# Backend Model Reference

This directory serves as the models placeholder for the backend service.

To avoid duplicating large binary model files:
- The backend defaults to reading the trained model from `../../ml-model/models/best_model.pth`.
- The class taxonomy is read from `../../ml-model/dataset/class_indices.json`.
- In a containerized or standalone production deployment, model weights can be placed directly into this directory or specified via the `MODEL_PATH` environment variable.
