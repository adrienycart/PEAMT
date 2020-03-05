library(incon)
library(reticulate)
use_python("/c/Users/Marco/Anaconda3/python")


folders <- list.dirs('features/chords_and_times', recursive=FALSE)
systems <- c('kelz', 'lisu', 'google', 'cheng', 'target')
models <- c("hutch_78_roughness", "har_18_harmonicity", "har_19_corpus")

py_save_object <- function(object, filename, pickle = "pickle", ...) {
    builtins <- import_builtins()
    pickle <- import(pickle)
    handle <- builtins$open(filename, "wb")
    on.exit(handle$close(), add = TRUE)
    pickle$dump(object, handle, protocol = pickle$HIGHEST_PROTOCOL, ...)
}

weighted_std <- function(values, weighted_mean, weights) {
    sigma <- 0.0
    for (i in 1:length(values)) {
        sigma <- sigma + ((values[i] - weighted_mean) ^ 2) * weights[i]
    }
    std <- sqrt(sigma / sum(weights))
    return(std)
}

for (folder in folders) {
    example <- substring(folder, 27)
    print(example)
    for (system in systems) {

        chords <- read.csv(paste(folder, "/", system, "_chords.csv", sep=""))
        durations <- read.csv(paste(folder, "/", system, "_durations.csv", sep=""))
        durations <- as.vector(t(durations[1]))
        # event_times <- read.csv(paste(folder, "/", system, "_event_times.csv", sep=""))
        
        hutch_78_roughness <- c()
        har_18_harmonicity <- c()
        har_19_corpus <- c()
        # loop over all the chords
        empty_chord_indexs <- c()
        for (idx in 1:dim(chords)[1]) {
            chord <- as.vector(t(chords[idx,]))
            chord <- chord[chord!=-1] # unpad
            duration <- durations[idx]

            if (length(chord) > 0) {
                consenance_values <- incon(chord, models)
                hutch_78_roughness <- append(hutch_78_roughness, unname(consenance_values[1]))
                har_18_harmonicity <- append(har_18_harmonicity, unname(consenance_values[2]))
                har_19_corpus <- append(har_19_corpus, unname(consenance_values[3]))
            } else {
                empty_chord_indexs <- append(empty_chord_indexs, idx)
                # consenance_values <- incon(c(90), models)
                # hutch_78_roughness <- append(hutch_78_roughness, unname(consenance_values[1]))
                # har_18_harmonicity <- append(har_18_harmonicity, unname(consenance_values[2]))
                # har_19_corpus <- append(har_19_corpus, unname(consenance_values[3]))
            }

        }
        if (length(empty_chord_indexs) > 0) {
            durations <- durations[-(empty_chord_indexs)]
        }

        hutch_78_roughness_weighted_mean <- weighted.mean(hutch_78_roughness, durations)
        har_18_harmonicity_weighted_mean <- weighted.mean(har_18_harmonicity, durations)
        har_19_corpus_weighted_mean <- weighted.mean(har_19_corpus, durations)

        hutch_78_roughness_weighted_std <- weighted_std(hutch_78_roughness, hutch_78_roughness_weighted_mean, durations)
        har_18_harmonicity_weighted_std <- weighted_std(har_18_harmonicity, har_18_harmonicity_weighted_mean, durations)
        har_19_corpus_weighted_std <- weighted_std(har_19_corpus, har_19_corpus_weighted_mean, durations)

        consenance_statistics <- c(hutch_78_roughness_weighted_mean, har_18_harmonicity_weighted_mean, har_19_corpus_weighted_mean, hutch_78_roughness_weighted_std, har_18_harmonicity_weighted_std, har_19_corpus_weighted_std, max(hutch_78_roughness), max(har_18_harmonicity), max(har_19_corpus), min(hutch_78_roughness), min(har_18_harmonicity), min(har_19_corpus))

        path = paste("features/consonance_statistics_no_empty_chords/", example, sep="")
        if (!file.exists("features/consonance_statistics_no_empty_chords")) {
            dir.create("features/consonance_statistics_no_empty_chords")
        }
        if (!file.exists(path)) {
            dir.create(path)
        }

        py_save_object(consenance_statistics, paste("features/consonance_statistics_no_empty_chords/", example, "/", system, ".pkl", sep=""))

    }
}


