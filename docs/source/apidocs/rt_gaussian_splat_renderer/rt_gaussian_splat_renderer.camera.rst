:py:mod:`rt_gaussian_splat_renderer.camera`
===========================================

.. py:module:: rt_gaussian_splat_renderer.camera

.. autodoc2-docstring:: rt_gaussian_splat_renderer.camera
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`cam_ray_gen <rt_gaussian_splat_renderer.camera.cam_ray_gen>`
     - .. autodoc2-docstring:: rt_gaussian_splat_renderer.camera.cam_ray_gen
          :summary:

API
~~~

.. py:function:: cam_ray_gen(extrinsic: taichi.math.mat4, intrinsic: taichi.math.mat3, res: typing.Tuple[int, int]) -> ti.field(Ray)
   :canonical: rt_gaussian_splat_renderer.camera.cam_ray_gen

   .. autodoc2-docstring:: rt_gaussian_splat_renderer.camera.cam_ray_gen
