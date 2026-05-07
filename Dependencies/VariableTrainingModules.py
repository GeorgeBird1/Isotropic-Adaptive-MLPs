from torch import nn
import torch
import numpy as np

def _build_variable_batch(batch_dictionary, dataset_blocks, total_input_width, device):
    combined_inputs = []
    batch_slices = {}
    label_dictionary = {}
    batch_start = 0

    for dataset_name, (batch_data, batch_labels) in batch_dictionary.items():
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        batch_size = batch_data.shape[0]
        flat_batch = batch_data.flatten(start_dim=1)
        full_input = torch.zeros(
            (batch_size, total_input_width),
            dtype=batch_data.dtype,
            device=device,
        )

        input_start, input_end, _, _ = dataset_blocks[dataset_name]
        full_input[:, input_start:input_end] = flat_batch

        combined_inputs.append(full_input)
        batch_slices[dataset_name] = slice(batch_start, batch_start + batch_size)
        label_dictionary[dataset_name] = batch_labels
        batch_start += batch_size

    return torch.cat(combined_inputs, dim=0), label_dictionary, batch_slices


def _variable_block_loss_and_accuracy(prediction, label_dictionary, batch_slices, dataset_blocks, loss, loss_weights=None):
    total_error = 0.0
    active_dataset_count = 0
    loss_dictionary = {}
    weighted_loss_dictionary = {}
    accuracy_dictionary = {}

    for dataset_name, labels in label_dictionary.items():
        _, _, output_start, output_end = dataset_blocks[dataset_name]
        logits = prediction[batch_slices[dataset_name], output_start:output_end]

        if isinstance(loss, nn.CrossEntropyLoss):
            block_error = loss(logits, labels)
        elif isinstance(loss, nn.MSELoss):
            target = torch.nn.functional.one_hot(
                labels, num_classes=output_end - output_start
            ).to(logits.dtype)
            block_error = loss(logits, target)
        else:
            raise TypeError(
                "For classification, loss must be nn.CrossEntropyLoss() or nn.MSELoss()"
            )

        block_error_value = float(block_error.detach().cpu().numpy())
        if loss_weights is None:
            block_weight = 1.0
        else:
            block_weight = float(loss_weights[dataset_name])

        weighted_block_error = block_weight * block_error
        weighted_block_error_value = float(weighted_block_error.detach().cpu().numpy())

        total_error = total_error + weighted_block_error
        if block_weight != 0.0:
            active_dataset_count += 1

        loss_dictionary[dataset_name] = block_error_value
        weighted_loss_dictionary[dataset_name] = weighted_block_error_value
        accuracy_dictionary[dataset_name] = 100.0 * (
            torch.argmax(logits, dim=1) == labels
        ).float().mean().item()

    if active_dataset_count > 0:
        total_error = total_error / active_dataset_count
    else:
        total_error = 0.0 * prediction.sum()

    return total_error, loss_dictionary, weighted_loss_dictionary, accuracy_dictionary


def testing_epoch_variable(network, testing_sets, dataset_blocks, total_input_width, device, loss, lambda_psi=0.0,
                           loss_weights=None):
    network.eval()
    test_cost = []
    test_loss = {dataset_name: [] for dataset_name in testing_sets}
    test_weighted_loss = {dataset_name: [] for dataset_name in testing_sets}
    test_acc = {dataset_name: [] for dataset_name in testing_sets}

    iterators = {dataset_name: iter(loader) for dataset_name, loader in testing_sets.items()}
    total_steps = min(len(loader) for loader in testing_sets.values())

    with torch.no_grad():
        for step in range(total_steps):
            batch_dictionary = {
                dataset_name: next(iterators[dataset_name])
                for dataset_name in testing_sets
            }

            combined_input, label_dictionary, batch_slices = _build_variable_batch(
                batch_dictionary=batch_dictionary,
                dataset_blocks=dataset_blocks,
                total_input_width=total_input_width,
                device=device,
            )

            prediction = network(combined_input)
            error, loss_dictionary, weighted_loss_dictionary, accuracy_dictionary = _variable_block_loss_and_accuracy(
                prediction=prediction,
                label_dictionary=label_dictionary,
                batch_slices=batch_slices,
                dataset_blocks=dataset_blocks,
                loss=loss,
                loss_weights=loss_weights,
            )

            if getattr(network, "linear_correction_approach", "").upper() == "TRAINABLE+DECAY":
                psi_decay = 0.0
                for psi in network.psi_parameters:
                    psi_decay = psi_decay + psi.square().sum()
                error = error + lambda_psi * psi_decay

            test_cost.append(float(error.detach().cpu().numpy()))
            for dataset_name, loss_value in loss_dictionary.items():
                test_loss[dataset_name].append(loss_value)
            for dataset_name, weighted_loss_value in weighted_loss_dictionary.items():
                test_weighted_loss[dataset_name].append(weighted_loss_value)
            for dataset_name, accuracy_value in accuracy_dictionary.items():
                test_acc[dataset_name].append(accuracy_value)

    test_cost = np.mean(test_cost)
    test_loss = {
        dataset_name: float(np.mean(dataset_loss))
        for dataset_name, dataset_loss in test_loss.items()
    }
    test_weighted_loss = {
        dataset_name: float(np.mean(dataset_loss))
        for dataset_name, dataset_loss in test_weighted_loss.items()
    }
    test_acc = {
        dataset_name: float(np.mean(dataset_accuracy))
        for dataset_name, dataset_accuracy in test_acc.items()
    }
    return network, test_cost, test_acc, test_loss, test_weighted_loss


def training_epoch_variable(network, training_sets, dataset_blocks, total_input_width, device, optimiser, loss,
                            current_epoch, lambda_psi=0.0, loss_weights=None):
    network.train()
    train_x = []
    train_cost = []
    train_loss = {dataset_name: [] for dataset_name in training_sets}
    train_weighted_loss = {dataset_name: [] for dataset_name in training_sets}
    train_acc = {dataset_name: [] for dataset_name in training_sets}

    iterators = {dataset_name: iter(loader) for dataset_name, loader in training_sets.items()}
    total_steps = min(len(loader) for loader in training_sets.values())

    for i in range(total_steps):
        fractional_epoch = current_epoch + i / total_steps
        optimiser.zero_grad(set_to_none=True)

        batch_dictionary = {
            dataset_name: next(iterators[dataset_name])
            for dataset_name in training_sets
        }

        combined_input, label_dictionary, batch_slices = _build_variable_batch(
            batch_dictionary=batch_dictionary,
            dataset_blocks=dataset_blocks,
            total_input_width=total_input_width,
            device=device,
        )

        prediction = network(combined_input)
        error, loss_dictionary, weighted_loss_dictionary, accuracy_dictionary = _variable_block_loss_and_accuracy(
            prediction=prediction,
            label_dictionary=label_dictionary,
            batch_slices=batch_slices,
            dataset_blocks=dataset_blocks,
            loss=loss,
            loss_weights=loss_weights,
        )

        if getattr(network, "linear_correction_approach", "").upper() == "TRAINABLE+DECAY":
            psi_decay = 0.0
            for psi in network.psi_parameters:
                psi_decay = psi_decay + psi.square().sum()
            error = error + lambda_psi * psi_decay

        train_x.append(fractional_epoch)
        train_cost.append(float(error.detach().cpu().numpy()))
        for dataset_name, loss_value in loss_dictionary.items():
            train_loss[dataset_name].append(loss_value)
        for dataset_name, weighted_loss_value in weighted_loss_dictionary.items():
            train_weighted_loss[dataset_name].append(weighted_loss_value)
        for dataset_name, accuracy_value in accuracy_dictionary.items():
            train_acc[dataset_name].append(accuracy_value)

        error.backward()
        optimiser.step()

    print("Done!")
    train_loss = {
        dataset_name: np.asarray(dataset_loss, dtype=np.float64)
        for dataset_name, dataset_loss in train_loss.items()
    }
    train_weighted_loss = {
        dataset_name: np.asarray(dataset_loss, dtype=np.float64)
        for dataset_name, dataset_loss in train_weighted_loss.items()
    }
    train_acc = {
        dataset_name: np.asarray(dataset_accuracy, dtype=np.float64)
        for dataset_name, dataset_accuracy in train_acc.items()
    }
    return network, train_x, train_cost, train_acc, train_loss, train_weighted_loss
