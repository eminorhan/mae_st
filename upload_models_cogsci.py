from huggingface_hub import HfApi

api = HfApi()

# upload model files

# kinetics
api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models/new/kinetics/kinetics.pth",
    path_in_repo="kinetics_none.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_ssv2-50shot.pth",
    path_in_repo="kinetics_ssv2-50shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_ssv2-10shot.pth",
    path_in_repo="kinetics_ssv2-10shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_kinetics-50shot.pth",
    path_in_repo="kinetics_kinetics-50shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_kinetics-10shot.pth",
    path_in_repo="kinetics_kinetics-10shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

#######################################################################################

# kinetics-200h
api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models/new/kinetics_128/kinetics_128.pth",
    path_in_repo="kinetics-200h_none.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_128_ssv2-50shot.pth",
    path_in_repo="kinetics-200h_ssv2-50shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_128_ssv2-10shot.pth",
    path_in_repo="kinetics-200h_ssv2-10shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_128_kinetics-50shot.pth",
    path_in_repo="kinetics-200h_kinetics-50shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_128_kinetics-10shot.pth",
    path_in_repo="kinetics-200h_kinetics-10shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

#######################################################################################

# say
api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models/new/say/say.pth",
    path_in_repo="say_none.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/say_ssv2-50shot.pth",
    path_in_repo="say_ssv2-50shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/say_ssv2-10shot.pth",
    path_in_repo="say_ssv2-10shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/say_kinetics-50shot.pth",
    path_in_repo="say_kinetics-50shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/say_kinetics-10shot.pth",
    path_in_repo="say_kinetics-10shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

#######################################################################################

# s
api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models/new/s/s.pth",
    path_in_repo="s_none.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/s_ssv2-50shot.pth",
    path_in_repo="s_ssv2-50shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/s_ssv2-10shot.pth",
    path_in_repo="s_ssv2-10shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/s_kinetics-50shot.pth",
    path_in_repo="s_kinetics-50shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/s_kinetics-10shot.pth",
    path_in_repo="s_kinetics-10shot.pth",
    repo_id="eminorhan/video-models",
    repo_type="model",
    token=True
)