# Mount only stage-required PPIFlow volumes

PPIFlow stage-specific remote wrappers mount only the workflow and app volumes needed by that stage. Shared helpers receive an explicit volume map from the wrapper, keeping mount scope smaller and making expected-file verification depend on declared stage inputs rather than every possible app volume.
