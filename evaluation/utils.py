def logger(log, tag):
    def emit(line):
        print(f"{tag} {line}", flush=True)
        log.write(f"{tag} {line}\n")
        log.flush()
    return emit


def patient_progress(emit):
    def on_patient(done, total, row):
        code, label, prob, per_image = row
        emit(f"[{done:>4}/{total}] {code}  images {len(per_image):>2}  "
             f"label {label}  prob_chd {prob:.4f}")
    return on_patient
