"""TTT."""
from ScanImageTiffReader import ScanImageTiffReader


class ScanImageTiffFile(ScanImageTiffReader):
    """TIFFILE.

    Wrapper around ScanImageTiffReader to make it more pythonic:
    - `shape` and `dtype` are now properties, not functions
    - description and metadata are parsed into dictionaries not raw strings
    - added filename attribute

    (link to ScanImageTiffReader and scanimage docs)

    Constructor:
        `file = ScanImageTiffFile(filename)`

    Supports context manager:
    ```
        with ScanImageTiffFile(filename) as f:
            do_something()
    ```
    Methods:
        data(beg=None, end=None)

    Attrs:
        name
        dtype
        shape
        description
        metadata
    """

    def __init__(self, filename):
        """Init."""
        super().__init__(filename)
        self.name = filename
        self.description = self._parse_description()
        self.metadata = self._parse_metadata()
        self.shape = self.shape()
        self.dtype = self.dtype()

    def _translate_matlab_to_python(self, val):
        """Translate a matlab string to python code."""
        # booleans
        val = val.replace('true', 'True')
        val = val.replace('false', 'False')
        # nan
        val = val.replace('NaN', 'np.nan')
        # lists - this messes up 2D matlab arrays
        if val[0] == '[':
            val = val.replace(' ', ',')
            val = val.replace(';', ',')
        try:
            val = eval(val)
        except (ValueError, NameError, SyntaxError):
            pass
        return val

    def _parse_description(self):
        """Parse the per-frame description embedded in tif-file to a dict.

        Returns:
            description as dict
        """
        nb_frames = self.shape()[0]
        description = dict()
        for cnt in range(nb_frames):
            tmp = self.description(cnt)
            lines = tmp.strip().split('\n')
            for line in lines:
                key, val = line.split(' = ')
                if key not in description:
                    description[key] = []  # init
                description[key].append(self._translate_matlab_to_python(val))
        return description

    def _parse_metadata(self):
        """Parse the meta data embedded in tif-file to a dict.

        Returns:
            metadata as dict
        """
        meta_lines = self.metadata().strip().split('\n')
        metadata = dict()
        for meta_line in meta_lines:
            try:
                key, val = meta_line.split(' = ')
                key = '.'.join(key.split('.')[1:])
                metadata[key] = self._translate_matlab_to_python(val)
            except (TypeError, ValueError, NameError, SyntaxError):
                pass
        return metadata
