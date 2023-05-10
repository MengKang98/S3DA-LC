settings = {
    "office-31": {
        "max_iter": 10000,
        "val_after": 100,
        "batch_size": 31,
        "val_batch_size": 100,
        "label_num": 31,
    },
    "office-home": {
        "max_iter": 30000,
        "val_after": 1000,
        "batch_size": 32,
        "val_batch_size": 1500,
        "label_num": 65,
    },
    "domain-net": {
        "max_iter": 300000,
        "val_after": 5000,
        "batch_size": 24,
        "val_batch_size": 1500,
        "label_num": 345,
    },
}

settings["office-31"]["DW_A"] = {"pretrain_iter": 1000}
settings["office-31"]["AD_W"] = {"pretrain_iter": 1000}
settings["office-31"]["AW_D"] = {"pretrain_iter": 1000}

settings["office-home"]["ACP_R"] = {"pretrain_iter": 4000}
settings["office-home"]["ACR_P"] = {"pretrain_iter": 10000}
settings["office-home"]["APR_C"] = {"pretrain_iter": 4000}
settings["office-home"]["CPR_A"] = {"pretrain_iter": 4000}

settings["domain-net"]["CIPQR_S"] = {"pretrain_iter": 80000}
settings["domain-net"]["CIPQS_R"] = {"pretrain_iter": 80000}
settings["domain-net"]["CIPSR_Q"] = {"pretrain_iter": 60000}
settings["domain-net"]["CPQRS_I"] = {"pretrain_iter": 60000}
settings["domain-net"]["CIQRS_P"] = {"pretrain_iter": 60000}
settings["domain-net"]["IPQRS_C"] = {"pretrain_iter": 100000}

settings["office-31"]["DW_A"]["src"] = ["dslr", "webcam"]
settings["office-31"]["AW_D"]["src"] = ["amazon", "webcam"]
settings["office-31"]["AD_W"]["src"] = ["amazon", "dslr"]

settings["office-31"]["DW_A"]["trgt"] = "amazon"
settings["office-31"]["AW_D"]["trgt"] = "dslr"
settings["office-31"]["AD_W"]["trgt"] = "webcam"

settings["office-home"]["ACP_R"]["src"] = ["Art", "Clipart", "Product"]
settings["office-home"]["ACR_P"]["src"] = ["Art", "Clipart", "Real_World"]
settings["office-home"]["APR_C"]["src"] = ["Art", "Product", "Real_World"]
settings["office-home"]["CPR_A"]["src"] = ["Clipart", "Product", "Real_World"]

settings["office-home"]["ACP_R"]["trgt"] = "Real_World"
settings["office-home"]["ACR_P"]["trgt"] = "Product"
settings["office-home"]["APR_C"]["trgt"] = "Clipart"
settings["office-home"]["CPR_A"]["trgt"] = "Art"

settings["domain-net"]["CIPQS_R"]["src"] = [
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "sketch",
]
settings["domain-net"]["CIPSR_Q"]["src"] = [
    "clipart",
    "infograph",
    "painting",
    "sketch",
    "real",
]
settings["domain-net"]["CPQRS_I"]["src"] = [
    "clipart",
    "painting",
    "quickdraw",
    "real",
    "sketch",
]
settings["domain-net"]["CIPQR_S"]["src"] = [
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
]
settings["domain-net"]["CIQRS_P"]["src"] = [
    "clipart",
    "infograph",
    "quickdraw",
    "real",
    "sketch",
]
settings["domain-net"]["IPQRS_C"]["src"] = [
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
]

settings["domain-net"]["CIPQS_R"]["trgt"] = "real"
settings["domain-net"]["CIPSR_Q"]["trgt"] = "quickdraw"
settings["domain-net"]["CPQRS_I"]["trgt"] = "infograph"
settings["domain-net"]["CIPQR_S"]["trgt"] = "sketch"
settings["domain-net"]["CIQRS_P"]["trgt"] = "painting"
settings["domain-net"]["IPQRS_C"]["trgt"] = "clipart"
