import numpy as np
import h5py
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LogNorm
import math
import os

# --------------------------------------------------------------
# ML Functions
# --------------------------------------------------------------
class OverlapRemovalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        inputs: a tensor of shape (N, 16, 3) with the following ordering per event:
           0-5:   jets
           6-8:   electrons
           9-11:  muons
           12-14: photons
           15:    MET (untouched)
        """
        def process_event(event):
            # Split the event into objects based on the new ordering.
            jets = event[0:6]       # shape (6, 3)
            electrons = event[6:9]  # shape (3, 3)
            muons = event[9:12]     # shape (3, 3)
            photons = event[12:15]  # shape (3, 3)
            met = event[15:16]      # shape (1, 3)

            # Helper: When an object is "removed" we zero its features.
            # For overlap checks, removed objects should not affect dR calculations.
            # The helper function below replaces removed objects (pt == 0)
            # with sentinel values (eta,phi = 1e6) so that distances computed with them are huge.
            def mask_removed(objs):
                # objs: shape (n, 3) with col0: pt, col1: eta, col2: phi.
                active = tf.expand_dims(objs[:, 0] > 0, axis=-1)  # (n,1)
                sentinel = tf.constant([0.0, 1e6, 1e6], dtype=objs.dtype)
                sentinel = tf.broadcast_to(sentinel, tf.shape(objs))
                return tf.where(active, objs, sentinel)

            # Helper: dphi (accounts for periodicity)
            def dphi(phi1, phi2):
                diff = phi1 - phi2
                return tf.math.floormod(diff + math.pi, 2 * math.pi) - math.pi

            # Helper: pairwise dR calculation between two sets of objects.
            def pairwise_dR(objs1, objs2):
                objs1 = mask_removed(objs1)
                objs2 = mask_removed(objs2)
                eta1 = objs1[:, 1]  # shape (N,)
                phi1 = objs1[:, 2]
                eta2 = objs2[:, 1]  # shape (M,)
                phi2 = objs2[:, 2]
                deta = tf.expand_dims(eta1, axis=1) - tf.expand_dims(eta2, axis=0)  # (N, M)
                dphi_val = dphi(tf.expand_dims(phi1, axis=1), tf.expand_dims(phi2, axis=0))  # (N, M)
                return tf.sqrt(deta**2 + dphi_val**2)  # (N, M)

            # Sequential update functions: each rule zeroes out objects that fail the overlap test.

            # RULE 1: Muon vs Electron: if any electron is within dR < 0.2 of a muon, remove that muon.
            def rule1_update(muons, electrons):
                dr = pairwise_dR(electrons, muons)  # shape (n_elec, n_muon)
                remove = tf.reduce_any(dr < 0.2, axis=0)  # for each muon
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(muons), muons)
            muons = rule1_update(muons, electrons)

            # RULE 2: Photon vs Electron: if any electron is within dR < 0.4 of a photon, remove that photon.
            def rule2_update(photons, electrons):
                dr = pairwise_dR(electrons, photons)  # shape (n_elec, n_photon)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(photons), photons)
            photons = rule2_update(photons, electrons)

            # RULE 3: Photon vs Muon: if any muon is within dR < 0.4 of a photon, remove that photon.
            def rule3_update(photons, muons):
                dr = pairwise_dR(muons, photons)  # shape (n_muon, n_photon)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(photons), photons)
            photons = rule3_update(photons, muons)

            # RULE 4: Jet vs Electron: if any electron is within dR < 0.2 of a jet, remove that jet.
            def rule4_update(jets, electrons):
                dr = pairwise_dR(electrons, jets)  # shape (n_elec, n_jet)
                remove = tf.reduce_any(dr < 0.2, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(jets), jets)
            jets = rule4_update(jets, electrons)

            # RULE 5: Electron vs Jet: if any jet is within dR < 0.4 of an electron, remove that electron.
            def rule5_update(electrons, jets):
                dr = pairwise_dR(jets, electrons)  # shape (n_jet, n_elec)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(electrons), electrons)
            electrons = rule5_update(electrons, jets)

            # RULE 6: Jet vs Muon: if any muon is within dR < 0.2 of a jet, remove that jet.
            def rule6_update(jets, muons):
                dr = pairwise_dR(muons, jets)  # shape (n_muon, n_jet)
                remove = tf.reduce_any(dr < 0.2, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(jets), jets)
            jets = rule6_update(jets, muons)

            # RULE 7: Muon vs Jet: if any jet is within dR < 0.4 of a muon, remove that muon.
            def rule7_update(muons, jets):
                dr = pairwise_dR(jets, muons)  # shape (n_jet, n_muon)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(muons), muons)
            muons = rule7_update(muons, jets)

            # RULE 8: Photon vs Jet: if any jet is within dR < 0.4 of a photon, remove that photon.
            def rule8_update(photons, jets):
                dr = pairwise_dR(jets, photons)  # shape (n_jet, n_photon)
                remove = tf.reduce_any(dr < 0.4, axis=0)
                return tf.where(tf.expand_dims(remove, axis=-1), tf.zeros_like(photons), photons)
            photons = rule8_update(photons, jets)

            # Reassemble the event in the new order: jets, electrons, muons, photons, MET.
            output_event = tf.concat([jets, electrons, muons, photons, met], axis=0)
            return output_event

        outputs = tf.map_fn(process_event, inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        return config



class DuplicateRemovalLayer(tf.keras.layers.Layer):
    def __init__(self, duplicate_threshold=0.05, **kwargs):
        """
        Args:
          duplicate_threshold (float): dR threshold below which the later object is considered a duplicate.
        """
        super().__init__(**kwargs)
        self.duplicate_threshold = duplicate_threshold

    def call(self, inputs):
        """
        inputs: a tensor of shape (N, 16, 3) with the following ordering per event:
           0-5:   jets
           6-8:   electrons
           9-11:  muons
           12-14: photons
           15:    MET (untouched)
        """
        def process_event(event):
            # Split the event into objects.
            jets = event[0:6]       # shape (6, 3)
            electrons = event[6:9]  # shape (3, 3)
            muons = event[9:12]     # shape (3, 3)
            photons = event[12:15]  # shape (3, 3)
            met = event[15:16]      # shape (1, 3)

            # Helper: For duplicate removal, we want to ignore objects that have already been removed.
            # If an object is removed (pt==0), we replace its (eta,phi) with sentinel values so that it does not spuriously match.
            def mask_removed(objs):
                # objs: shape (n, 3) with columns: [pt, eta, phi]
                active = tf.expand_dims(objs[:, 0] > 0, axis=-1)  # shape (n,1)
                sentinel = tf.constant([0.0, 1e6, 1e6], dtype=objs.dtype)
                sentinel = tf.broadcast_to(sentinel, tf.shape(objs))
                return tf.where(active, objs, sentinel)

            # Helper: Compute pairwise dR for objects of the same type.
            def pairwise_dR_same(objs):
                # First mask out removed objects.
                objs_masked = mask_removed(objs)
                eta = objs_masked[:, 1]  # shape (n,)
                phi = objs_masked[:, 2]  # shape (n,)
                deta = tf.expand_dims(eta, axis=1) - tf.expand_dims(eta, axis=0)  # (n, n)
                dphi_val = tf.math.floormod(tf.expand_dims(phi, axis=1) - tf.expand_dims(phi, axis=0) + math.pi, 
                                            2 * math.pi) - math.pi  # (n, n)
                return tf.sqrt(deta**2 + dphi_val**2)  # (n, n)

            # Duplicate removal: For objects of the same type, if two objects are within duplicate_threshold,
            # we remove (zero out) the one with the higher index.
            def remove_duplicates(objs, threshold):
                # objs: shape (n, 3)
                n = tf.shape(objs)[0]
                # If there are no objects, just return.
                # (tf.cond is not strictly necessary here since n is small, but we can check if desired.)
                # Compute pairwise distances.
                dr = pairwise_dR_same(objs)  # shape (n, n)
                # Create a lower-triangular mask (excluding the diagonal) so that for each object j, we consider only objects i < j.
                ones = tf.ones_like(dr, dtype=tf.bool)
                # tf.linalg.band_part(ones, 0, -1) gives the upper-triangular part (including diagonal).
                # Its logical_not gives the strictly lower-triangular part.
                lower_tri_exclusive = tf.logical_not(tf.linalg.band_part(ones, 0, -1))
                # For positions where the mask is False, assign a large value so they don't affect our check.
                replaced_dr = tf.where(lower_tri_exclusive, dr, tf.fill(tf.shape(dr), 1e6))
                # For each column j, if any entry in rows 0..j-1 is below the threshold, mark object j as duplicate.
                duplicates = tf.reduce_any(replaced_dr < threshold, axis=0)  # shape (n,)
                # Zero out duplicate objects.
                new_objs = tf.where(tf.expand_dims(duplicates, axis=-1), tf.zeros_like(objs), objs)
                return new_objs

            # Now apply duplicate removal to each object type (except MET).
            jets = remove_duplicates(jets, self.duplicate_threshold)
            electrons = remove_duplicates(electrons, self.duplicate_threshold)
            muons = remove_duplicates(muons, self.duplicate_threshold)
            photons = remove_duplicates(photons, self.duplicate_threshold)

            # Reassemble the event in the original order.
            output_event = tf.concat([jets, electrons, muons, photons, met], axis=0)
            return output_event

        outputs = tf.map_fn(process_event, inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'duplicate_threshold': self.duplicate_threshold})
        return config



# class ReorderObjectsLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def call(self, inputs):
#         """
#         inputs: a tensor of shape (N, 16, 3) with the following ordering per event:
#            0-5:   jets
#            6-8:   electrons
#            9-11:  muons
#            12-14: photons
#            15:    MET (untouched)
#         """
#         def process_event(event):
#             # Split the event into its constituent object collections.
#             jets = event[0:6]       # shape (6, 3)
#             electrons = event[6:9]  # shape (3, 3)
#             muons = event[9:12]     # shape (3, 3)
#             photons = event[12:15]  # shape (3, 3)
#             met = event[15:16]      # shape (1, 3)

#             #Helper function that reorders a collection so that non-zero objects (pt > 0)
#             #come first, in their original order, followed by zeroed objects.
#             def reorder_collection(collection):
#                 # Determine which objects are non-zero (based on pt in column 0).
#                 non_zero_mask = collection[:, 0] > 0
#                 non_zero = tf.boolean_mask(collection, non_zero_mask)
#                 zeros = tf.boolean_mask(collection, tf.logical_not(non_zero_mask))
#                 return tf.concat([non_zero, zeros], axis=0)

#             # def reorder_collection(collection):
#             #     condition = tf.reduce_all(tf.equal(collection, 0))
#             #     return tf.cond(
#             #         condition,
#             #         lambda: collection,  # if true, just return collection
#             #         lambda: _do_reorder(collection)  # otherwise, perform the reordering
#             #     )

#             # def _do_reorder(collection):
#             #     non_zero_mask = collection[:, 0] > 0
#             #     non_zero = tf.boolean_mask(collection, non_zero_mask)
#             #     zeros = tf.boolean_mask(collection, tf.logical_not(non_zero_mask))
#             #     return tf.concat([non_zero, zeros], axis=0)

    

#             # Reorder each collection individually.
#             jets = reorder_collection(jets)
#             electrons = reorder_collection(electrons)
#             muons = reorder_collection(muons)
#             photons = reorder_collection(photons)

#             output_event = tf.concat([jets, electrons, muons, photons, met], axis=0)
#             tf.debugging.assert_equal(tf.shape(output_event)[0], 16, message="Per-event shape must be 16 rows")
#             # tf.debugging.assert_equal(tf.shape(output_event)[0], 16)
#             # tf.print("Output event shape:", tf.shape(output_event))
#             #output_event.set_shape([16, 3])

#             # Reassemble the event in the original overall ordering.
#             return output_event

#         # Apply the reordering to each event in the batch.
#         outputs = tf.map_fn(process_event, inputs)
#         return outputs

#     def get_config(self):
#         config = super().get_config()
#         return config



class ReorderObjectsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Expected sizes per object group.
        self.expected = {
            'jets': 6,
            'electrons': 3,
            'muons': 3,
            'photons': 3,
            'met': 1
        }

    def call(self, inputs):
        """
        inputs: a tensor of shape (N, 16, 3) with the following ordering per event:
           0-5:   jets
           6-8:   electrons
           9-11:  muons
           12-14: photons
           15:    MET (untouched)
        """
        def pad_collection(collection, expected_size):
            # collection: tensor of shape (n, features)
            # Select objects with pt > 0.
            non_zero = tf.boolean_mask(collection, collection[:, 0] > 0)
            # Truncate if there are more than expected_size.
            non_zero = non_zero[:expected_size]
            # Determine how many rows we got.
            k = tf.shape(non_zero)[0]
            pad_size = expected_size - k
            # Create padding of zeros.
            zeros = tf.zeros((pad_size, tf.shape(collection)[1]), dtype=collection.dtype)
            padded = tf.concat([non_zero, zeros], axis=0)
            # Force the static shape.
            padded.set_shape([expected_size, collection.shape[1]])
            return padded

        def process_event(event):
            # Slice the event into groups.
            jets = event[0:6]       # shape (6, 3)
            electrons = event[6:9]  # shape (3, 3)
            muons = event[9:12]     # shape (3, 3)
            photons = event[12:15]  # shape (3, 3)
            met = event[15:16]      # shape (1, 3) – assumed fixed
            
            # Process each collection to guarantee fixed size.
            jets = pad_collection(jets, self.expected['jets'])
            electrons = pad_collection(electrons, self.expected['electrons'])
            muons = pad_collection(muons, self.expected['muons'])
            photons = pad_collection(photons, self.expected['photons'])
            # MET is assumed to be already fixed.
            
            # Concatenate the groups.
            output_event = tf.concat([jets, electrons, muons, photons, met], axis=0)
            # Now enforce that each event is exactly 16 rows.
            output_event.set_shape([16, event.shape[1]])
            return output_event

        outputs = tf.map_fn(process_event, inputs)
        # Optionally enforce the batch shape if known, e.g. (None, 16, 3)
        outputs.set_shape([None, 16, inputs.shape[-1]])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'expected': self.expected})
        return config


class DeltaPhiPreprocessingLayer(tf.keras.layers.Layer):
    def call(self, data):
        phi = data[:, :, 2]
        pt = data[:, :, 0]

        leading_jet_phi = phi[:, 0]
        pi = tf.constant(np.pi, dtype=tf.float32)  # Fix for tf.pi
        dphi = tf.math.mod(phi - tf.expand_dims(leading_jet_phi, axis=-1) + pi, 2 * pi) - pi

        zeroed_mask = tf.equal(pt, 0)
        phi_transformed = tf.where(zeroed_mask, tf.zeros_like(dphi), dphi)

        data_transformed = tf.concat([data[:, :, :2], tf.expand_dims(phi_transformed, axis=-1)], axis=-1)
        return data_transformed

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape


class METBiasMaskLayer(tf.keras.layers.Layer):
    def call(self, data):
        MET_values = data[:, -1, :]

        MET_zeros = tf.equal(MET_values[:, 0], 0)
        MET_neg999 = tf.equal(MET_values[:, 0], -999)
        MET_nan = tf.math.is_nan(MET_values[:, 2])

        MET_values = tf.where(tf.expand_dims(MET_zeros, axis=-1), tf.constant([[0, 0, 0]], dtype=data.dtype), MET_values)
        MET_values = tf.where(tf.expand_dims(MET_neg999 | MET_nan, axis=-1), tf.zeros_like(MET_values), MET_values)

        data_transformed = tf.concat([data[:, :-1, :], tf.expand_dims(MET_values, axis=1)], axis=1)
        return data_transformed



class ZeroOutLowPtLayer(tf.keras.layers.Layer):
    def __init__(self, pt_thresholds, **kwargs):
        super().__init__(**kwargs)
        self.pt_thresholds = pt_thresholds

    def call(self, data):
        jet_mask = tf.expand_dims(data[:, :6, 0] < self.pt_thresholds[0], axis=-1)
        electron_mask = tf.expand_dims(data[:, 6:9, 0] < self.pt_thresholds[1], axis=-1)
        muon_mask = tf.expand_dims(data[:, 9:12, 0] < self.pt_thresholds[2], axis=-1)
        photon_mask = tf.expand_dims(data[:, 12:15, 0] < self.pt_thresholds[3], axis=-1)

        data = tf.concat([
            tf.where(jet_mask, tf.zeros_like(data[:, :6, :]), data[:, :6, :]),
            tf.where(electron_mask, tf.zeros_like(data[:, 6:9, :]), data[:, 6:9, :]),
            tf.where(muon_mask, tf.zeros_like(data[:, 9:12, :]), data[:, 9:12, :]),
            tf.where(photon_mask, tf.zeros_like(data[:, 12:15, :]), data[:, 12:15, :]),
            data[:, 15:, :]
        ], axis=1)
        return data



class NormalizePtLayer(tf.keras.layers.Layer):
    def __init__(self, scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, data):
        data_scaled = tf.concat([
            tf.expand_dims(data[:, :, 0] * self.scale_factor, axis=-1),
            data[:, :, 1:]
        ], axis=-1)
        return data_scaled


class ScalePtPerEvent(tf.keras.layers.Layer):
    def __init__(self, target_sum=10.0, epsilon=1e-6, **kwargs):
        """
        Scales the pt's for each event so that the sum of the pt's equals target_sum.
        """
        super().__init__(**kwargs)
        self.target_sum = target_sum
        self.epsilon = epsilon

    def call(self, inputs):

        pts = inputs[:, :, 0]           # Shape: (N, 16)
        other_features = inputs[:, :, 1:]  # Shape: (N, 16, 2)

        # Compute the sum of pts per event (resulting shape: (N, 1))
        sum_pts = tf.reduce_sum(pts, axis=1, keepdims=True)

        # Compute the per-event scaling factor so that new sum equals target_sum.
        scale = self.target_sum / (sum_pts + self.epsilon)  # Shape: (N, 1)

        # Scale the pt's
        scaled_pts = pts * scale  # Still shape: (N, 16)

        # Expand dims to match the other features along the last axis.
        scaled_pts = tf.expand_dims(scaled_pts, axis=-1)  # Shape: (N, 16, 1)

        # Concatenate the scaled pts with the other features
        outputs = tf.concat([scaled_pts, other_features], axis=-1)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_sum': self.target_sum,
            'epsilon': self.epsilon,
        })
        return config


class MSEADScoreLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        y, x = inputs
        mask = tf.logical_and(tf.not_equal(y, 0), tf.not_equal(y, -999))
        mask = tf.cast(mask, dtype=tf.float32)
        _y = y * mask
        _x = x * mask
        squared_diff = tf.square(_y - _x)
        sum_squared_diff = tf.reduce_sum(squared_diff, axis=-1)
        valid_count = tf.reduce_sum(mask, axis=-1)
        valid_count = tf.where(valid_count == 0, 1.0, valid_count)
        mse = sum_squared_diff / valid_count
        return mse



def create_large_AE_with_preprocessed_inputs(
    num_objects, num_features, h_dim_1, h_dim_2, h_dim_3, h_dim_4, latent_dim, 
    pt_thresholds, scale_factor, l2_reg=0.01, dropout_rate=0, pt_normalization_type='global_division', overlap_removal=False, duplicate_removal=False
):
    # Preprocessing Layers
    # add a initial zero out layer with [30, 15, 15, 15]
    overlap_removal_layer = OverlapRemovalLayer()
    duplicate_removal_layer = DuplicateRemovalLayer()
    reorder_objects_layer = ReorderObjectsLayer()
    phi_rotation_layer = DeltaPhiPreprocessingLayer()
    met_bias_layer = METBiasMaskLayer()
    zero_out_layer = ZeroOutLowPtLayer(pt_thresholds)
    if pt_normalization_type == 'global_division':
        normalize_pt_layer = NormalizePtLayer(scale_factor)
    elif pt_normalization_type == 'per_event':
        normalize_pt_layer = ScalePtPerEvent(target_sum=10.0)
    flatten_layer = tf.keras.layers.Flatten()

    # Preprocessing Model
    preprocessing_inputs = layers.Input(shape=(num_objects * num_features,))
    unflattened = tf.keras.layers.Reshape((num_objects, num_features))(preprocessing_inputs)
    preprocessed = phi_rotation_layer(unflattened)

    if duplicate_removal:
        preprocessed = duplicate_removal_layer(preprocessed)

    if overlap_removal:
        preprocessed = overlap_removal_layer(preprocessed)

    if duplicate_removal or overlap_removal:
        preprocessed = reorder_objects_layer(preprocessed)
    
    preprocessed = met_bias_layer(preprocessed)
    preprocessed = zero_out_layer(preprocessed)
    preprocessed = normalize_pt_layer(preprocessed)
    preprocessed_flattened = flatten_layer(preprocessed)

    preprocessing_model = tf.keras.Model(inputs=preprocessing_inputs, outputs=preprocessed_flattened)

    # Encoder (takes preprocessed input)
    encoder_inputs = layers.Input(shape=(num_objects * num_features,))  # Preprocessed input
    x = layers.Dense(h_dim_1, kernel_regularizer=regularizers.l2(l2_reg))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_2, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_3, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_4, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    z = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)

    encoder = tf.keras.Model(inputs=encoder_inputs, outputs=z)

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h_dim_4, kernel_regularizer=regularizers.l2(l2_reg))(decoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_3, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_2, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_1, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_objects * num_features, kernel_regularizer=regularizers.l2(l2_reg))(x)
    decoder = tf.keras.Model(inputs=decoder_inputs, outputs=outputs)

    # Autoencoder (works directly on preprocessed input)
    ae_inputs = layers.Input(shape=(num_objects * num_features,))  # Preprocessed input
    reconstructed = decoder(encoder(ae_inputs))  # Encode and decode
    autoencoder = tf.keras.Model(inputs=ae_inputs, outputs=reconstructed)

    # MSE Model
    mse_scores = MSEADScoreLayer()([ae_inputs, reconstructed])  # Compare preprocessed input to reconstructed output
    mse_ae_model = tf.keras.Model(inputs=ae_inputs, outputs=mse_scores)

    return autoencoder, encoder, decoder, mse_ae_model, preprocessing_model

def loss_fn(y_true, y_pred):
    """Masked MSE with correct averaging by the number of valid objects."""
    
    # Masks to filter out invalid objects (zero padding and -999 placeholder)
    mask0 = K.cast(K.not_equal(y_true, 0), K.floatx())
    maskMET = K.cast(K.not_equal(y_true, -999), K.floatx())

    # Mask to upweight the first 6 elements (first two jets)
    weight = 1
    weight_mask = tf.ones_like(y_true)
    weight_mask = tf.concat([tf.ones_like(y_true[:, :6]) * weight, 
                      tf.ones_like(y_true[:, 6:])], 1)
    
    mask = mask0 * maskMET
    
    # Apply the mask to the squared differences
    squared_difference = K.square(mask * (y_pred - y_true)) * weight_mask
    
    # Sum the squared differences and the mask (to count valid objects)
    sum_squared_difference = K.sum(squared_difference, 1)
    valid_count = K.sum(mask, 1)  # Number of valid objects
    
    # Replace 0s by 1s
    valid_count = tf.where(K.equal(valid_count, 0), tf.ones_like(valid_count), valid_count)
    
    # Calculate the mean squared error by dividing by the number of valid objects
    mean_squared_error = sum_squared_difference / valid_count
    
    # Return the mean over the batch
    return K.mean(mean_squared_error)


def initialize_model(input_dim, pt_thresholds=[0,0,0,0], pt_scale_factor=0.05, dropout_p=0, L2_reg_coupling=0, latent_dim=4, saved_model_path=None, save_version=None, obj_type='HLT', pt_normalization_type='global_division', overlap_removal=False, duplicate_removal=False):
    '''
    Inputs:
        save_path: string of the path to save the model.
        dropout_p: dropout percentage for the AE.
        L2_reg_coupling: coupling value for L2 regularization.
        latent_dim: dimension of the latent space of the model.
        large_network: boolean for whether the network should be large or small.
        saved_model_path: None or string. If string, loads the weights from the saved model path.
        save_version: None or string. If string, suffix of the model to be loaded.

    Returns:
        HLT_AE: full autoencoder model to be used with HLT objects
        HLT_encoder: just the encoder of HLT_AE
        L1_AE: full autoencoder model to be used with L1 objects
        L1_encoder: just the encoder of L1_AE
    '''

    # Initialize models
    INPUT_DIM = input_dim
    H_DIM_1 = 100
    H_DIM_2 = 100
    H_DIM_3 = 64
    H_DIM_4 = 32
    LATENT_DIM = latent_dim
        
    HLT_AE, HLT_encoder, HLT_decoder, HLT_MSE_AE, HLT_preprocessing_model = create_large_AE_with_preprocessed_inputs(
        num_objects=16, 
        num_features=3, 
        h_dim_1=H_DIM_1, 
        h_dim_2=H_DIM_2, 
        h_dim_3=H_DIM_3, 
        h_dim_4=H_DIM_4, 
        latent_dim=LATENT_DIM,
        pt_thresholds=pt_thresholds,
        scale_factor=pt_scale_factor,
        l2_reg=L2_reg_coupling, 
        dropout_rate=dropout_p,
        pt_normalization_type=pt_normalization_type,
        overlap_removal=overlap_removal,
        duplicate_removal=duplicate_removal
    )
    # -------------------

    # Compile
    optimizer = Adam(learning_rate=0.001)
    HLT_AE.compile(optimizer=optimizer, loss=loss_fn, weighted_metrics=[])
    # -------------------

    # Load model weights (if specified in the args)
    if (saved_model_path is None) != (save_version is None):
        raise ValueError("Either both or neither of 'saved_model_path' and 'save_version' should be None.")
        
    if (saved_model_path is not None) and (save_version is not None):
        HLT_AE.load_weights(f'{saved_model_path}/EB_{obj_type}_HLT_{save_version}.weights.h5')
        HLT_encoder.load_weights(f'{saved_model_path}/EB_{obj_type}_HLT_encoder_{save_version}.weights.h5')
        HLT_MSE_AE.load_weights(f'{saved_model_path}/EB_{obj_type}_MSE_HLT_AE_{save_version}.weights.h5')
        HLT_preprocessing_model.load_weights(f'{saved_model_path}/EB_{obj_type}_preprocessing_A{save_version}.weights.h5')
    # -------------------

    return HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model

def HLT_load_and_inference(data, save_version=4):

    training_info = {
        "save_path": "/eos/home-m/mmcohen/ad_trigger_development/trained_models/trial_136", 
        "dropout_p": 0.1, 
        "L2_reg_coupling": 0.01, 
        "latent_dim": 4, 
        "large_network": False, 
        "num_trainings": 10, 
        "training_weights": True, 
        "obj_type": "HLT", 
        "overlap_removal": False,
        "duplicate_removal": False
    }

    data_info = {
        "train_data_scheme": "topo2A_train", 
        "pt_normalization_type": "global_division", 
        "L1AD_rate": 1000, 
        "pt_thresholds": [50, 30, 30, 30], 
        "pt_scale_factor": 0.05, 
        "comments": "",
        "plots_path": "/eos/home-m/mmcohen/ad_trigger_development/trained_models/trial_136/136-4_vnice_plots"
    }


    HLT_AE, HLT_encoder, HLT_MSE_AE, HLT_preprocessing_model = initialize_model(
        input_dim=data.shape[1],
        pt_thresholds=data_info['pt_thresholds'],
        pt_scale_factor=data_info['pt_scale_factor'],
        dropout_p=training_info['dropout_p'],
        L2_reg_coupling=training_info['L2_reg_coupling'],
        latent_dim=training_info['latent_dim'],
        saved_model_path=training_info['save_path'],
        save_version=save_version,
        obj_type='HLT',
        pt_normalization_type=data_info['pt_normalization_type'],
        overlap_removal=training_info['overlap_removal'],
        duplicate_removal=training_info['duplicate_removal']
    )

    # Preprocess the data
    prep_data = HLT_preprocessing_model.predict(data, verbose=0, batch_size=8)

    # Calculate the latent representations
    z = HLT_encoder.predict(prep_data, batch_size=8, verbose=0)

    # Calculate the AD scores
    AD_scores = HLT_MSE_AE.predict(prep_data, batch_size=8, verbose=0)

    return AD_scores, z


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

def make_plots(data_dict, reference_data_dict, pt_thresholds=[0, 999999], require_pass_jet_trigger=False):
    plt.figure(figsize=(15, 8))
    data_mask = (data_dict['HLT_data'][:, 0] > pt_thresholds[0]) & (data_dict['HLT_data'][:, 0] < pt_thresholds[1])
    reference_data_mask = (reference_data_dict['HLT_data'][:, 0] > pt_thresholds[0]) & (reference_data_dict['HLT_data'][:, 0] < pt_thresholds[1])
    bins = np.linspace(0, 50, 35)
    if require_pass_jet_trigger:
        data_mask = data_mask & (data_dict['pass_single_jet_trigger'])
        print(f"Number of data events with single jet trigger: {np.sum(data_mask)}")
        reference_data_mask = reference_data_mask & (reference_data_dict['pass_single_jet_trigger'])
        print(f"Number of reference data events with single jet trigger: {np.sum(reference_data_mask)}")
        bins = np.linspace(0, 50, 20)
    plt.hist(data_dict['HLTAD_scores'][data_mask], bins=bins, density=True, histtype='step', linewidth=2.5, fill=False, label='data25')
    plt.hist(reference_data_dict['HLTAD_scores'][reference_data_mask], bins=bins, density=True, histtype='step', linewidth=2.5, fill=False, label='reference data24')
    plt.xlabel('HLT AD Score')
    plt.yscale('log')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.title(f'HLT AD Scores with range {pt_thresholds[0]} GeV - {pt_thresholds[1]} GeV')
    plt.savefig('HLT_scores.png')
    plt.close()

    plt.figure(figsize=(15, 8))
    data_mask = (data_dict['L1_data'][:, 0] > pt_thresholds[0]) & (data_dict['L1_data'][:, 0] < pt_thresholds[1])
    reference_data_mask = (reference_data_dict['L1_data'][:, 0] > pt_thresholds[0]) & (reference_data_dict['L1_data'][:, 0] < pt_thresholds[1])
    if require_pass_jet_trigger:
        data_mask = data_mask & (data_dict['pass_single_jet_trigger'])
        reference_data_mask = reference_data_mask & (reference_data_dict['pass_single_jet_trigger'])
    plt.hist(data_dict['L1AD_scores'][data_mask], bins=bins, density=True, histtype='step', linewidth=2.5, fill=False, label='data25')
    plt.hist(reference_data_dict['L1AD_scores'][reference_data_mask], bins=bins, density=True, histtype='step', linewidth=2.5, fill=False, label='reference data24')
    plt.xlabel('L1 AD Score')
    plt.yscale('log')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.title(f'L1 AD Scores with range {pt_thresholds[0]} GeV - {pt_thresholds[1]} GeV')
    plt.savefig('L1_scores.png')
    plt.close()

def inference(ntuple_file):

    L1AD_threshold = 7.1248 # 1000 Hz unique rate
    HLTAD_threshold = 7.9862 # 10 Hz unique rate

    # load the data
    data_dict = {}
    with h5py.File(ntuple_file, 'r') as hf:
        for key in hf.keys():
            data_dict[key] = hf[key][:]


    # Run the inference
    HLT_AD_scores, HLT_z = HLT_load_and_inference(data_dict['HLT_data'], save_version=4)

    data_dict['HLTAD_scores'] = HLT_AD_scores
    data_dict['HLT_z'] = HLT_z


    # Assign trigger decisions based on the AD scores, including seeding from L1AD
    data_dict['pass_L1AD'] = data_dict['L1AD_scores'] > L1AD_threshold # L1AD sees all events
    data_dict['pass_HLTAD'] = (data_dict['HLTAD_scores'] > HLTAD_threshold) & (data_dict['pass_L1AD']) # HLTAD sees only events that pass L1AD



    # Get the current date in YYYYMMDD format
    current_date = datetime.now().strftime("%Y%m%d")

    # save the data_dict to the same h5 file
    with h5py.File(ntuple_file, 'a') as hf:
        for key, value in data_dict.items():
            if key not in hf.keys():
                hf.create_dataset(key, data=value)
            else:
                del hf[key]
                hf.create_dataset(key, data=value)

    # Get run number from the run_numbers dataset in the ntuple file
    run_number = data_dict['run_numbers'][0]
    
    # Update output directory name to include run number
    output_dir = f"../results/{current_date}_{run_number}"
    os.makedirs(output_dir, exist_ok=True)


    


def plot(ntuple_file, pt_thresholds=[0, 999999], require_pass_jet_trigger=False):

    # load the data
    data_dict = {}
    with h5py.File(ntuple_file, 'r') as hf:
        for key in hf.keys():
            data_dict[key] = hf[key][:]

    # load the reference data
    reference_data_dict = {}
    with h5py.File('/eos/home-m/mmcohen/ad_trigger_development/ops/data/ntuples/data_dict_20250515_475341.h5', 'r') as hf:
        for key in hf.keys():
            reference_data_dict[key] = hf[key][:]

    # Make the plots
    make_plots(data_dict, reference_data_dict, pt_thresholds, require_pass_jet_trigger)

    # Move the plots and data_dict to the new directory
    run_number = data_dict['run_numbers'][0]
    current_date = datetime.now().strftime("%Y%m%d")
    output_dir = f"../results/{current_date}_{run_number}"
    os.rename('HLT_scores.png', os.path.join(output_dir, 'HLT_scores.png'))
    os.rename('L1_scores.png', os.path.join(output_dir, 'L1_scores.png'))