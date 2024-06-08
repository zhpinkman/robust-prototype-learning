import torch
from transformers import (
    BartForConditionalGeneration,
)
import torchmetrics

# torch.manual_seed(0)
# import random
# random.seed(0)
# Custom modules


class ProtoTEx(torch.nn.Module):
    def __init__(
        self,
        num_prototypes,
        n_classes,
        max_length,
        class_weights=None,
        bias=True,
        special_classfn=False,
        p=0.5,
        batchnormlp1=True,
        use_cosine_dist=False,
    ):
        super().__init__()

        self.bart_model = BartForConditionalGeneration.from_pretrained(
            "ModelTC/bart-base-mnli"
        )

        # self.bart_model = BartForConditionalGeneration.from_pretrained(
        #     "facebook/bart-large-mnli"
        # )
        self.n_classes = n_classes
        self.bart_out_dim = self.bart_model.config.d_model
        self.one_by_sqrt_bartoutdim = 1 / torch.sqrt(
            torch.tensor(self.bart_out_dim).float()
        )
        self.max_position_embeddings = max_length
        self.num_protos = num_prototypes
        self.use_cosine_dist = use_cosine_dist
        print("Using cosine distance: ", self.use_cosine_dist)

        self.prototypes = torch.nn.Parameter(
            torch.rand(self.num_protos, self.max_position_embeddings, self.bart_out_dim)
        )

        # TODO: Try setting bias to True
        self.classfn_model = torch.nn.Linear(self.num_protos, self.n_classes, bias=bias)

        #         self.loss_fn=torch.nn.BCEWithLogitsLoss(reduction="mean")
        if class_weights is not None:
            print("Using class weights for cross entropy loss...")
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.Tensor(class_weights), reduction="mean"
            )
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        #         self.set_encoder_status(False)
        #         self.set_decoder_status(False)
        #         self.set_protos_status(False)
        #         self.set_classfn_status(False)
        self.special_classfn = special_classfn

        self.dobatchnorm = (
            batchnormlp1  # This flag is actually for instance normalization
        )
        self.distance_grounder = torch.zeros(self.n_classes, self.num_protos).cuda()
        for i in range(self.n_classes):
            # self.distance_grounder[i][np.random.randint(0, self.num_protos, int(self.num_protos / 2))] = 1e7
            self.distance_grounder[i][self.num_protos :] = 1e7

    def set_prototypes(self, input_ids_rdm, attn_mask_rdm, do_random=False):
        if do_random:
            print("initializing prototypes with xavier init")
            torch.nn.init.xavier_normal_(self.prototypes)
        else:
            # Use this when the dataset is balanced (augmented)
            print("initializing prototypes with encoded outputs")
            self.eval()
            with torch.no_grad():
                self.prototypes = torch.nn.Parameter(
                    self.bart_model.base_model.encoder(
                        input_ids_rdm.cuda(),
                        attn_mask_rdm.cuda(),
                        output_attentions=False,
                        output_hidden_states=False,
                    ).last_hidden_state
                )

    def set_shared_status(self, status=True):
        self.bart_model.model.shared.requires_grad_(status)

    def set_encoder_status(self, status=True):
        self.num_enc_layers = len(self.bart_model.base_model.encoder.layers)

        for i in range(self.num_enc_layers):
            self.bart_model.base_model.encoder.layers[i].requires_grad_(False)
        self.bart_model.base_model.encoder.layers[
            self.num_enc_layers - 1
        ].requires_grad_(status)
        return

    def set_decoder_status(self, status=True):
        self.num_dec_layers = len(self.bart_model.base_model.decoder.layers)
        for i in range(self.num_dec_layers):
            self.bart_model.base_model.decoder.layers[i].requires_grad_(False)
        self.bart_model.base_model.decoder.layers[
            self.num_dec_layers - 1
        ].requires_grad_(status)
        return

    def set_classfn_status(self, status=True):
        self.classfn_model.requires_grad_(status)

    def set_protos_status(self, status=True):
        self.prototypes.requires_grad = status

    def predict(self, input_ids, attn_mask):
        batch_size = input_ids.size(0)
        last_hidden_state = self.bart_model.base_model.encoder(
            input_ids.cuda(),
            attn_mask.cuda(),
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state
        if not self.dobatchnorm:
            input_for_classfn = torch.cdist(
                last_hidden_state.view(batch_size, -1),
                self.prototypes.view(self.num_protos, -1),
            )
        else:
            # from IPython import embed
            try:
                # print(last_hidden_state.shape)
                # print(self.prototypes.shape)

                input_for_classfn = torch.cdist(
                    last_hidden_state.view(batch_size, -1),
                    self.prototypes.view(self.num_protos, -1),
                )
            except:
                exit()
            # embed()
            input_for_classfn = torch.nn.functional.instance_norm(
                input_for_classfn.view(batch_size, 1, self.num_protos)
            ).view(batch_size, self.num_protos)

        return self.classfn_model(input_for_classfn).view(batch_size, self.n_classes)

    def forward(
        self,
        input_ids,
        attn_mask,
        y,
        use_classfn=0,
        use_rc=0,
        use_p1=0,
        use_p2=0,
        use_p3=0,
        classfn_lamb=1.0,
        rc_loss_lamb=0.95,
        p1_lamb=0.93,
        p2_lamb=0.92,
        p3_lamb=1.0,
        distmask_lp1=0,
        distmask_lp2=0,
        random_mask_for_distanceMat=None,
    ):
        """
        1. p3_loss is the prototype-distance-maximising loss. See the set of lines after the line "if use_p3:"
        2. We also have flags distmask_lp1 and distmask_lp2 which uses "masked" distance matrix for calculating lp1 and lp2 loss.
        3. the flag "random_mask_for_distanceMat" is an experimental part. It randomly masks (artificially inflates)
        random places in the distance matrix so as to encourage more prototypes be "discovered" by the training
        examples.
        """
        batch_size = input_ids.size(0)

        rc_loss = torch.tensor(0)
        last_hidden_state = self.bart_model.base_model.encoder(
            input_ids.cuda(),
            attn_mask.cuda(),
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state

        # Lp3 is minimize the negative of inter-prototype distances (maximize the distance)
        input_for_classfn, l_p1, l_p2, l_p3, _, classfn_out, classfn_loss = (
            None,
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            None,
            torch.tensor(0),
        )
        if use_classfn or use_p1 or use_p2 or use_p3:
            all_protos = self.prototypes
            if use_classfn or use_p1 or use_p2:
                if not self.dobatchnorm:
                    # TODO: This loss function is not ignoring the padded part of the sequence; Get element-wise distane and then multiply with the mask
                    if self.use_cosine_dist:
                        input_for_classfn = (
                            torchmetrics.functional.pairwise_cosine_similarity(
                                last_hidden_state.view(batch_size, -1),
                                all_protos.view(self.num_protos, -1),
                            )
                        )
                    else:
                        input_for_classfn = torch.cdist(
                            last_hidden_state.view(batch_size, -1),
                            all_protos.view(self.num_protos, -1),
                        )

                else:
                    # TODO: Try cosine distance
                    if self.use_cosine_dist:
                        input_for_classfn = (
                            torchmetrics.functional.pairwise_cosine_similarity(
                                last_hidden_state.view(batch_size, -1),
                                all_protos.view(self.num_protos, -1),
                            )
                        )
                    else:
                        input_for_classfn = torch.cdist(
                            last_hidden_state.view(batch_size, -1),
                            all_protos.view(self.num_protos, -1),
                        )
                    input_for_classfn = torch.nn.functional.instance_norm(
                        input_for_classfn.view(batch_size, 1, self.num_protos)
                    ).view(batch_size, self.num_protos)
            if use_p1 or use_p2:
                # This part is for seggregating training of negative and positive prototypes
                distance_mask = self.distance_grounder[y.cuda()]
                input_for_classfn_masked = input_for_classfn + distance_mask
                if random_mask_for_distanceMat:
                    random_mask = torch.bernoulli(
                        torch.ones_like(input_for_classfn_masked)
                        * random_mask_for_distanceMat
                    ).bool()
                    input_for_classfn_masked[random_mask] = 1e7
        #                     print(input_for_classfn_masked)
        if use_p1:
            l_p1 = torch.mean(
                torch.min(
                    input_for_classfn_masked if distmask_lp1 else input_for_classfn,
                    dim=0,
                )[0]
            )
        if use_p2:
            l_p2 = torch.mean(
                torch.min(
                    input_for_classfn_masked if distmask_lp2 else input_for_classfn,
                    dim=1,
                )[0]
            )
        if use_p3:
            # Used for Inter-prototype distance
            #             l_p3 = self.one_by_sqrt_bartoutdim * torch.mean(torch.pdist(all_protos.view(self.num_protos,-1)))
            l_p3 = self.one_by_sqrt_bartoutdim * torch.mean(
                torch.pdist(self.prototypes.view(self.num_protos, -1))
            )
        if use_classfn:
            classfn_out = self.classfn_model(input_for_classfn).view(
                batch_size, self.n_classes
            )
            classfn_loss = self.loss_fn(classfn_out, y.cuda())
        if not use_rc:
            rc_loss = torch.tensor(0)
        total_loss = (
            classfn_lamb * classfn_loss
            + rc_loss_lamb * rc_loss
            + p1_lamb * l_p1
            + p2_lamb * l_p2
            - p3_lamb * l_p3
        )
        return classfn_out, (
            total_loss,
            classfn_loss.detach().cpu(),
            rc_loss.detach().cpu(),
            l_p1.detach().cpu(),
            l_p2.detach().cpu(),
            l_p3.detach().cpu(),
        )
