from train import Trainer, parse_args
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    args = parse_args()
    config = Trainer.load_config(args.config)

    det_loaders, clf_loaders, trans_loaders, det_class_names, clf_class_names = Trainer.build_data_loaders(config)
    model = Trainer.build_model(config, det_class_names, clf_class_names)

    trainer = Trainer(
        model=model,
        detector_loaders=det_loaders,
        classifier_loader=clf_loaders,
        translator_loaders=trans_loaders,
        config=config
    )

    model.eval()
    val_loader = det_loaders[1]
    with torch.no_grad():
        for images, targets in val_loader:
            images = [images[0].to(trainer.device)]
            outputs = model(images)
            print(outputs)
            # visualize the results
            for i, output in enumerate(outputs[0:1]):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                boxes = output['boxes'].cpu().numpy()
                meta_labels = output['meta_labels'].cpu().numpy()
                symbol_labels = output['symbol_labels'].cpu().numpy() 
                scores = output['scores'].cpu().numpy()
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    meta_class = det_class_names[meta_labels[j]]
                    symbol_class = clf_class_names[symbol_labels[j]]
                    score = scores[j]
                    # add class name above the box
                    ax.text(x1, y1 - 10, f'{meta_class}/{symbol_class}: {score:.2f}', color='yellow', fontsize=8, backgroundcolor='black')
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                # add title with the latex expression
                ax.set_title(f'Predicted LaTeX: {output.get("latex_expression", "")}', fontsize=10)
                plt.show()
            break