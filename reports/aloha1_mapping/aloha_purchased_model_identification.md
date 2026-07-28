# Purchased ALOHA Model Identification

- Status: `PASS`
- Classification: `SIMPLE_ALOHA_VIPER_2024_5_13_STEP`
- Confidence: `DIRECT_MODEL_AND_DIMENSION_MATCH`

The supplied engineering drawing directly names `Aloha ViperX 6DOF` and `Aloha VX300S Follower Robot Arm`. Its 204 x 299.46 mm base matches the Simple Viper AP214 geometry.

## First-hand source chain

- Sales/product page: `https://idminer.com.tw/product/aloha-viperx/`
- ViperX sales sheet: `https://drive.google.com/file/d/11KcnA49dhTiOD_MxmmC_SG75Cs97-JKh/view?usp=sharing`
- VX300S technical drawing: `https://drive.google.com/file/d/11M96-4JDw0y31OZMTQQ3Nqz1qCIqk_DU/view?usp=sharing`
- ALOHA 3D CAD folder: `https://drive.google.com/drive/folders/1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf`
- Trossen ALOHA manual: `https://docs.trossenrobotics.com/aloha_docs/`

| Candidate | CAD family | CAD base X x Y | Drawing error X x Y | Result |
|---|---|---:|---:|---|
| Simple Aloha Viper | VX | 204.000000 x 299.462987 mm | 0.000000 x 0.002987 mm | MATCH |
| Aloha Widow with Gripper | WX | 153.072000 x 233.536000 mm | 50.928000 x 65.924000 mm | NOT_THE_PURCHASED_FOLLOWER_ARM |

The two STEP files look similar at the gripper because both embed the same `Aloha VX Fingers 2024-4-21` pair with equal labels, topology, volumes, and pair bounds. That shared end effector does not make the WX/Widow arm a VX300S follower.
