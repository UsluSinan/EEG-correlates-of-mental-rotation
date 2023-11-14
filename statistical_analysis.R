# install and load required packages
for (package.name in c('MKinfer', 'afex', 'ggplot2')) {
    if (!require(package.name, character.only=TRUE)){
        install.packages(package.name, dependencies=TRUE)
        require(package.name, character.only=TRUE)
    }
}

# define global variables
plot_dir = './output/plots'
sink_path = './output/statistical_analysis.txt' 
dpi = 600
representative_participant = 'id000'

# direct output to file
sink()
sink(sink_path)

# read data
evaluation = read.csv('./output/modeling_results.csv')
features = read.csv('./output/features_importance.csv')
rts = read.csv('./output/rts.csv')

# aggregate MAE across participants
evaluation |>
    subset(measure=='MAE') |>
    (\(data){
        aggs = aggregate(
            value ~ testParticipant + model + testSetPreprocessor + angularDisparity,
            data=data,
            FUN=\(x) c(
                n=length(x),
                MAE.mean=mean(x), 
                MAE.sd=sd(x), 
                MAE.se=sd(x)/sqrt(length(x))
                )
            )
        for (dimname in dimnames(aggs$value)[[2]]){
            aggs[dimname] = aggs$value[, dimname]
        }
        aggs$value = NULL
        return(aggs)
        })() |>
    print()

# intraindividual evaluation for all angles
print('intra-individual evaluation for all angles')
evaluation |>
    subset(
        testParticipant=='same_as_trainParticipant' & 
        angularDisparity=='all' & 
        measure=='MAE',
        select=-c(
            testParticipant, 
            testSetPreprocessor, 
            angularDisparity, 
            n_trainSamples, 
            n_testSamples,
            measure
            )
        ) |>
    reshape(
        direction='wide',
        idvar='trainParticipant',
        timevar='model'
        ) |>
    with(
        boot.t.test(
            value.eeg,
            value.rt,
            paired=TRUE
            )
        ) |>
    print()

# intraindividual evaluation per angle
print('intra-individual evaluation per angle')
evaluation |>
    subset(
        testParticipant=='same_as_trainParticipant' & 
        model=='eeg' & 
        angularDisparity!='all' & 
        measure=='MAE'
        ) |>
    (\(data){
        aov_res = aov_car(
            value ~ angularDisparity + Error(trainParticipant/angularDisparity),
            data=data
            )
        return(aov_res)
        })() |>
    summary() |>
    print()

print("inter-individual evaluation: trainParticipant's preprocessors")
evaluation |> 
    subset(
        model == 'eeg' &
        testSetPreprocessor == 'all_from_trainParticipant' &
        angularDisparity == 'all' &
        measure == 'MAE'
        ) |>
    reshape(
        direction='wide',
        idvar='trainParticipant',
        timevar='testParticipant'
        ) |>
    with(
        boot.t.test(
            value.same_as_trainParticipant,
            value.other_participant,
            paired=TRUE
            )
        ) |>
    print()

print("inter-individual evaluation: within other participant")
evaluation |>
    subset(
        model == 'eeg' &
        testParticipant == 'other_participant' &
        testSetPreprocessor %in% c('all_from_trainParticipant', 'all_from_testParticipant') &
        measure == 'MAE' &
        angularDisparity == 'all'
        ) |>
    reshape(
        direction='wide',
        idvar='trainParticipant',
        timevar='testSetPreprocessor'
        ) |>
    with(
        boot.t.test(
            value.all_from_trainParticipant,
            value.all_from_testParticipant,
            paired=TRUE
            )
        ) |>
    print()

print("inter-individual evluation: with corresponding pre-processors")
evaluation |>
    (\(data){
        intra.evaluation = data |>
            subset(
                model == 'eeg' &
                testParticipant == 'same_as_trainParticipant' &
                testSetPreprocessor == 'all_from_trainParticipant' &
                measure == 'MAE' &
                angularDisparity == 'all'
                )
        inter.evaluation = data |>
            subset(
                model == 'eeg' &
                testParticipant == 'other_participant' &
                testSetPreprocessor == 'all_from_testParticipant' &
                measure == 'MAE' &
                angularDisparity == 'all'
                )
        selected.evaluation = rbind(
            intra.evaluation, 
            inter.evaluation
            )
        return(selected.evaluation)
        })() |>
        reshape(
            direction='wide',
            idvar='trainParticipant',
            timevar='testParticipant'
            ) |>
        with(
            boot.t.test(
                value.same_as_trainParticipant,
                value.other_participant,
                paired=TRUE
                )
            ) |>
        print()

# plot intraindividual evaluation
evaluation |>
    subset(
        testParticipant=='same_as_trainParticipant' &
        angularDisparity=='all' &
        measure=='MAE'
        ) |>
    ggplot(aes(x=model, y=value)) +
    stat_boxplot(geom='errorbar', width=.25) +
    geom_boxplot(width=.5) +
    scale_x_discrete(labels=c('EEG', 'RT')) +
    labs(x='model', y='MAE') +
        theme(
        panel.grid=element_line(colour='#D3D3D3'),
        panel.background=element_blank(),
        panel.border=element_rect(fill=NA),
        plot.title = element_text(hjust=.5, size=20),
        plot.margin=grid::unit(c(1, 5, 0, 1), 'mm'),
        axis.text = element_text(size=8),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12)
        )

ggsave(
    filename='intraindividual_evluation.tif',
    device='tiff',
    compression='lzw',
    path=plot_dir,
    width=3,
    height=4,
    dpi=dpi
    )


# plot feature importance scores for representative participant
features |>
    (\(data){
        data.sub = subset(data, participant==representative_participant)
        data.sub$feature.order = 0
        ordered_features = unique(data$feature)
        for (idx in seq_along(ordered_features)){
            data.sub[data.sub$feature == ordered_features[idx], 'feature.order'] = idx
        }
        data.sub = within(data.sub, feature <- reorder(feature, feature.order))
        return(data.sub)
        })() |>
    ggplot(aes(x=feature, y=mean_abs_shap)) +
    geom_bar(
        stat='identity', 
        color='#999999',
        fill='#999999',
        alpha=.5
        ) +
    labs(x='frequency band', y='mean |SHAP value|') +
    theme(
        panel.grid=element_line(colour='#D3D3D3'),
        panel.background=element_blank(),
        panel.border=element_rect(fill=NA),
        plot.title = element_text(hjust=.5, size=20),
        plot.margin=grid::unit(c(1, 5, 0, 1), 'mm'),
        axis.title = element_text(size=18),
        axis.text = element_text(size=12),
        legend.title = element_text(size=12),
        legend.text = element_text(size=10)
        ) +
    coord_flip()

ggsave(
    filename=paste0('feature_importance_', representative_participant, '.tif'),
    device='tiff',
    compression='lzw',
    path=plot_dir,
    width=7,
    height=6,
    dpi=dpi
    )

# plot aggregated feature importance scores
features |>
    (\(data){

        aggs = aggregate(
            mean_abs_shap ~ feature, 
            data, 
            \(x) c(
                y=median(x),
                ymin=median(x) - mad(x, constant=1),
                ymax=median(x) + mad(x, constant=1)
                )
            )
        for (dimname in dimnames(aggs$mean_abs_shap)[[2]]){
            aggs[dimname] = aggs$mean_abs_shap[, dimname]
        }
        aggs$mean_abs_shap = NULL

        # order features
        aggs$feature.order = 0
        ordered_features = unique(data$feature)
        for (idx in seq_along(ordered_features)){
            aggs[aggs$feature == ordered_features[idx], 'feature.order'] = idx
        }
        aggs = within(aggs, feature <- reorder(feature, feature.order))
        return(aggs)
        })() |>

    ggplot(aes(x=feature, y=y)) +
    geom_bar(
        stat='identity', 
        color='#999999', 
        fill='#999999', 
        alpha=.5
        ) +
    geom_errorbar(
        aes(ymin=ymin, ymax=ymax),
        width=.2,
        #position=position_dodge(.9)
        ) +
    labs(y='median |SHAP value|') +
    theme(
        panel.grid=element_line(colour='#D3D3D3'),
        plot.margin=grid::unit(c(7, 5, 0, 6), 'mm'),
        panel.background=element_blank(),
        panel.border=element_rect(fill=NA),
        text=element_text(family='sans', size=10),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=10),
        axis.text.y = element_text(size=8),
        axis.text.x = element_text(size=8, angle=90, hjust=1, vjust=.5),
        legend.title = element_text(size=10),
        legend.text = element_text(size=10)
    )

ggsave(
    filename='feature_importance_aggregated.tif',
    device='tiff',
    compression='lzw',
    path=plot_dir,
    width=5,
    height=7,
    dpi=dpi
    )

# plot heat map of feature associations
ordered_features = unique(features$feature)
feature_associations = expand.grid(
    feature.1=ordered_features,
    feature.2=ordered_features,
    score=0
    )
for (feat.1 in feature_associations$feature.1){
    for (feat.2 in feature_associations$feature.2){
        feat.1.score = with(features, mean_abs_shap[feature==feat.1])
        feat.2.score = with(features, mean_abs_shap[feature==feat.2])
        similarity = mean(abs(feat.1.score - feat.2.score))
        this_association = with(feature_associations, feature.1 == feat.1 & feature.2 == feat.2)
        feature_associations[this_association, 'score'] = similarity
    }
}
feature_associations |>
    ggplot(aes(x=feature.1, y=feature.2, fill=score)) +
    geom_tile() +
    scale_fill_viridis_c() +
    scale_x_discrete(labels=ordered_features) +
    scale_y_discrete(labels=ordered_features) +
    coord_cartesian(ylim=c(1, 10), xlim=c(1, 10), clip="off") +
    theme(
        text=element_text(family='sans'),
        panel.grid=element_line(colour='#D3D3D3'),
        plot.margin=grid::unit(c(7, 9, 1, 15), 'mm'),
        panel.background=element_blank(),
        panel.border=element_rect(fill=NA),
        axis.title = element_blank(),
        axis.text.y = element_text(size=8),
        axis.text.x = element_text(size=8, angle=90, hjust=1, vjust=.5),
        legend.title = element_text(size=10),
        legend.text = element_text(size=8)
    )

ggsave(
    filename='feature_associations.tif',
    device='tiff',
    compression='lzw',
    path=plot_dir,
    width=7,
    height=7,
    dpi=dpi
    )

# plot RTs before and after log-transformation
rts |>
    (\(data){
        means = aggregate(rt ~ angle + unit, data, mean)
        p = ggplot(data, aes(x=rt)) +
            geom_histogram(
                data=subset(data, unit=='ms'), 
                color='#999999', 
                fill='#999999', 
                binwidth = 750, 
                position='identity', 
                alpha=.5
                ) +
            geom_histogram(
                data=subset(data, unit=='log_ms'), 
                color='#999999', 
                fill='#999999', 
                binwidth = .35, 
                position='identity', 
                alpha=.5
                ) +
            geom_vline(
                data=means, 
                aes(xintercept=rt),
                linetype='dashed',
                color='#999999'
                ) +
            facet_grid(
                angle ~ factor(unit, levels=c('ms', 'log_ms')),
                scales='free_x',
                labeller=as_labeller(
                    c(
                        '0' = 'angle: 0°',
                        '50' = 'angle: 50°',
                        '100' = 'angle: 100°',
                        '150' = 'angle: 150°',
                        'log_ms' = 'unit: log(ms)',
                        'ms' = 'unit: ms'
                        )
                    )
                ) +
            xlab('RT') +
            theme(
                panel.grid=element_line(colour='#D3D3D3'),
                panel.background=element_blank(),
                panel.border=element_rect(fill=NA),
                text=element_text(family='sans', size=10),
                axis.title = element_text(size=12),
                axis.text = element_text(size=8),
                legend.text = element_text(size=10)
                )
            return(p)
        })()

ggsave(
    filename='rts_pre-post_log-transformation.tif',
    device='tiff',
    compression='lzw',
    path=plot_dir,
    width=5,
    height=6,
    dpi=dpi
    )

#sink()
unlink(sink_path)
