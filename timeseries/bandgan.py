import torch
import torch.nn as nn
from utils.loss import GANLoss


class BandGan:
    def __init__(
        self,
        batch_size,
        seq_len,
        in_dim,
        device,
        dataloader,
        netD,
        netG,
        optimD,
        optimG,
        critic_iter=5,
    ):
        print("init")
        self.critic_iter = critic_iter
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.device = device
        self.dataloader = dataloader
        self.netD = (netD,)
        self.netG = (netG,)
        self.optimD = optimD
        self.optimG = optimG

    def train(self, epochs):
        device = self.device
        shape = (self.batch_size, self.seq_len, self.in_dim)
        criterion_adv = GANLoss(target_real_label=0.9, target_fake_label=0.1).to(device)
        criterion_l1n = nn.SmoothL1Loss().to(device)
        criterion_l2n = nn.MSELoss().to(device)

        for epoch in range(epochs):
            for i in range(self.critic_iter):
                z = torch.randn(shape).to(device)
                fake_batch, real_batch, start_prices = self.data.get_samples(
                    G=self.G,
                    latent_dim=self.latent_dim,
                    n=self.batch_size,
                    ts_dim=self.ts_dim,
                    conditional=self.conditional,
                    use_cuda=self.use_cuda,
                )
                print("critic", i)


class BandGan2:
    def __init__(
        self,
        shape,
        encoder_input_shape,
        generator_input_shape,
        critic_x_input_shape,
        critic_z_input_shape,
        layers_encoder,
        layers_generator,
        layers_critic_x,
        layers_critic_z,
        optimizer,
        learning_rate=0.0005,
        epochs=2000,
        latent_dim=20,
        batch_size=64,
        iterations_critic=5,
        **hyperparameters
    ):

        self.shape = shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.iterations_critic = iterations_critic
        self.epochs = epochs
        self.hyperparameters = hyperparameters

        self.encoder_input_shape = encoder_input_shape
        self.generator_input_shape = generator_input_shape
        self.critic_x_input_shape = critic_x_input_shape
        self.critic_z_input_shape = critic_z_input_shape

        self.layers_encoder, self.layers_generator = layers_encoder, layers_generator
        self.layers_critic_x, self.layers_critic_z = layers_critic_x, layers_critic_z

        # self.optimizer = import_object(optimizer)(learning_rate)

    def _build_tadgan(self, **kwargs):

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        self.encoder = self._build_model(
            hyperparameters, self.layers_encoder, self.encoder_input_shape
        )
        self.generator = self._build_model(
            hyperparameters, self.layers_generator, self.generator_input_shape
        )
        self.critic_x = self._build_model(
            hyperparameters, self.layers_critic_x, self.critic_x_input_shape
        )
        self.critic_z = self._build_model(
            hyperparameters, self.layers_critic_z, self.critic_z_input_shape
        )

        self.generator.trainable = False
        self.encoder.trainable = False

        z = Input(shape=(self.latent_dim, 1))
        x = Input(shape=self.shape)
        x_ = self.generator(z)
        z_ = self.encoder(x)
        fake_x = self.critic_x(x_)
        valid_x = self.critic_x(x)
        interpolated_x = RandomWeightedAverage()([x, x_])
        validity_interpolated_x = self.critic_x(interpolated_x)
        partial_gp_loss_x = partial(
            self._gradient_penalty_loss, averaged_samples=interpolated_x
        )
        partial_gp_loss_x.__name__ = "gradient_penalty"
        self.critic_x_model = Model(
            inputs=[x, z], outputs=[valid_x, fake_x, validity_interpolated_x]
        )
        self.critic_x_model.compile(
            loss=[self._wasserstein_loss, self._wasserstein_loss, partial_gp_loss_x],
            optimizer=self.optimizer,
            loss_weights=[1, 1, 10],
        )

        fake_z = self.critic_z(z_)
        valid_z = self.critic_z(z)
        interpolated_z = RandomWeightedAverage()([z, z_])
        validity_interpolated_z = self.critic_z(interpolated_z)
        partial_gp_loss_z = partial(
            self._gradient_penalty_loss, averaged_samples=interpolated_z
        )
        partial_gp_loss_z.__name__ = "gradient_penalty"
        self.critic_z_model = Model(
            inputs=[x, z], outputs=[valid_z, fake_z, validity_interpolated_z]
        )
        self.critic_z_model.compile(
            loss=[self._wasserstein_loss, self._wasserstein_loss, partial_gp_loss_z],
            optimizer=self.optimizer,
            loss_weights=[1, 1, 10],
        )

        self.critic_x.trainable = False
        self.critic_z.trainable = False
        self.generator.trainable = True
        self.encoder.trainable = True

        z_gen = Input(shape=(self.latent_dim, 1))
        x_gen_ = self.generator(z_gen)
        x_gen = Input(shape=self.shape)
        z_gen_ = self.encoder(x_gen)
        x_gen_rec = self.generator(z_gen_)
        fake_gen_x = self.critic_x(x_gen_)
        fake_gen_z = self.critic_z(z_gen_)

        self.encoder_generator_model = Model(
            [x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec]
        )
        self.encoder_generator_model.compile(
            loss=[self._wasserstein_loss, self._wasserstein_loss, "mse"],
            optimizer=self.optimizer,
            loss_weights=[1, 1, 10],
        )
