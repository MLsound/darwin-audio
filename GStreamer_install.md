# Installation guide for GST Peaq & GStreamer
## This file offers a step by step guide for installing the GST Peaq plugin for the GStreamer framework

1. Download & Install GStreamer framework (development installer):

    ▸ https://gstreamer.freedesktop.org/download/

2. Clone GitHub repo:

    `git clone https://github.com/HSU-ANT/gstpeaq.git`

    `cd gstpeaq`

3. Follow installation guidelines:

    ▸ https://github.com/HSU-ANT/gstpeaq/blob/develop/INSTALL

4. Prerequisites:

    `brew install pkg-config autoconf automake libtool gtk-doc`

5. Create Log file:

    `touch ChangeLog`

6. Set Environment Variables:

    `export PKG_CONFIG_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib/pkgconfig:$PKG_CONFIG_PATH"`

    `export ACLOCAL_PATH="/usr/local/share/aclocal:$ACLOCAL_PATH"`

    `export PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/bin:$PATH"`

7. Disable documentation building:
    1. Open file:

        `nano Makefile.am`

    2. Find this line:

        `SUBDIRS = src doc tests`

        (or `SUBDIRS = src doc`)

    3. Change to:

        `SUBDIRS = src tests`

        (or `SUBDIRS = src`)

8. Configure the Build System:

    `autoreconf -vfi`

    `./configure --disable-gtk-doc --disable-man`

9. Compile & Install GstPEAQ:

    `make`

    `sudo make install`

10. Verification after installation:

    `gst-inspect-1.0 --scan`

    `gst-inspect-1.0 peaq`

## Ready to use GstPEAQ through CLI! 🎶

`peaq [--advanced] {REFFILE} {TESTFILE}`

Expected output:

`Objective Difference Grade: -2.161`

`Distortion Index: -0.269`