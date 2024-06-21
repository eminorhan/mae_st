from huggingface_hub import HfApi

api = HfApi()

# upload model files

# @ 224 px - kinetics
api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models/new/kinetics/kinetics.pth",
    path_in_repo="kinetics_none.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_ssv2-50shot.pth",
    path_in_repo="kinetics_ssv2-50shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_ssv2-10shot.pth",
    path_in_repo="kinetics_ssv2-10shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_kinetics-50shot.pth",
    path_in_repo="kinetics_kinetics-50shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/kinetics_kinetics-10shot.pth",
    path_in_repo="kinetics_kinetics-10shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned_imagenet/new/kinetics_imagenet_0.02.pth",
    path_in_repo="kinetics_imagenet-2pt.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

#######################################################################################

# @ 224 px - humanlike
api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models/new/sayavakepicutego4d/sayavakepicutego4d.pth",
    path_in_repo="hvm1_none.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/sayavakepicutego4d_ssv2-50shot.pth",
    path_in_repo="hvm1_ssv2-50shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/sayavakepicutego4d_ssv2-10shot.pth",
    path_in_repo="hvm1_ssv2-10shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/sayavakepicutego4d_kinetics-50shot.pth",
    path_in_repo="hvm1_kinetics-50shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/sayavakepicutego4d_kinetics-10shot.pth",
    path_in_repo="hvm1_kinetics-10shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned_imagenet/new/sayavakepicutego4d_imagenet_0.02.pth",
    path_in_repo="hvm1_imagenet-2pt.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

#######################################################################################
# @ 448 px - humanlike
api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models/new/sayavakepicutego4d_448/sayavakepicutego4d_448.pth",
    path_in_repo="hvm1@448_none.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/sayavakepicutego4d_448_ssv2-50shot.pth",
    path_in_repo="hvm1@448_ssv2-50shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/sayavakepicutego4d_448_ssv2-10shot.pth",
    path_in_repo="hvm1@448_ssv2-10shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/sayavakepicutego4d_448_kinetics-50shot.pth",
    path_in_repo="hvm1@448_kinetics-50shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned/new/sayavakepicutego4d_448_kinetics-10shot.pth",
    path_in_repo="hvm1@448_kinetics-10shot.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)

api.upload_file(
    path_or_fileobj="/scratch/eo41/mae_st/models_finetuned_imagenet/new/sayavakepicutego4d_448_imagenet_0.02.pth",
    path_in_repo="hvm1@448_imagenet-2pt.pth",
    repo_id="eminorhan/hvm-1",
    repo_type="model",
    token='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'
)